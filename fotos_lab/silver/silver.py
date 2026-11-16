from pyspark.sql import Window
from pyspark.sql import functions as F

from comum import caminho, criar_sessao, exigir_camada, glob_eventos, gravar_eventos, ler_eventos, titulo

CATALOGO = [
    "page_view", "scroll", "click", "site_search", "video_start", "video_progress", "video_complete",
    "file_download", "view_item_list", "select_item", "view_item", "add_to_cart", "remove_from_cart",
    "view_cart", "begin_checkout", "add_shipping_info", "add_payment_info", "purchase", "refund",
    "generate_lead", "sign_up", "login",
]
MARCOS_SCROLL = [25, 50, 75, 90]
MARCOS_VIDEO = [0, 10, 25, 50, 75, 100]
SESSAO_SEGUNDOS = 30 * 60

spark = criar_sessao("silver")
exigir_camada(spark, glob_eventos("bronze"), "python pipeline/bronze.py")

bronze = ler_eventos(spark, "bronze")
total_bronze = bronze.count()

ordem_dedup = Window.partitionBy("event_id").orderBy("data_ingestao", "arquivo_origem")
dedup = (bronze.withColumn("_n", F.row_number().over(ordem_dedup))
         .filter("_n = 1").drop("_n"))
total_dedup = dedup.count()

params = bronze.schema["params"].dataType.fieldNames()
eventos = dedup.select(
    "event_id", "event_name",
    F.col("event_ts").alias("event_ts_original"),
    F.col("event_ts").try_cast("timestamp").alias("event_ts"),
    "user_pseudo_id", "user_id",
    F.col("page.location").alias("page_location"),
    F.col("page.referrer").alias("page_referrer"),
    F.col("traffic.source").alias("traffic_source"),
    F.col("traffic.medium").alias("traffic_medium"),
    F.col("traffic.campaign").alias("traffic_campaign"),
    F.col("device.category").alias("device_category"),
    F.col("device.os").alias("device_os"),
    F.col("device.browser").alias("device_browser"),
    *[F.col(f"params.{p}").alias(p) for p in params],
    "items", "arquivo_origem", "data_ingestao", "year", "month", "day",
)

sem_transacao = F.col("transaction_id").isNull() | (F.trim("transaction_id") == "")
fora_marco_scroll = (F.col("event_name") == "scroll") & (
    F.col("percent_scrolled").isNull() | ~F.col("percent_scrolled").isin(MARCOS_SCROLL))
fora_marco_video = F.col("event_name").startswith("video_") & (
    F.col("video_percent").isNull() | ~F.col("video_percent").isin(MARCOS_VIDEO))

eventos = eventos.withColumn(
    "motivo_rejeicao",
    F.when(F.col("event_name").isNull() | ~F.col("event_name").isin(CATALOGO), "evento fora do catálogo")
    .when(F.col("event_ts").isNull() | (F.col("event_ts") > F.col("data_ingestao")), "timestamp ausente ou futuro")
    .when(F.col("event_name").isin("purchase", "refund") & sem_transacao, "transação sem id")
    .when((F.col("value") < 0) | (F.col("value").isNotNull() & (F.coalesce(F.col("currency"), F.lit("")) != "BRL")),
          "valor monetário inválido")
    .when(fora_marco_scroll | fora_marco_video, "marco fora do padrão"),
)

compras_validas = (eventos.filter((F.col("event_name") == "purchase") & F.col("motivo_rejeicao").isNull())
                   .select("transaction_id").distinct()
                   .withColumn("_compra_existe", F.lit(True)))

eventos = (eventos.join(compras_validas, "transaction_id", "left")
           .withColumn("motivo_rejeicao", F.when(
               F.col("motivo_rejeicao").isNull() & (F.col("event_name") == "refund") & F.col("_compra_existe").isNull(),
               "reembolso sem compra").otherwise(F.col("motivo_rejeicao")))
           .drop("_compra_existe"))
eventos.cache()

rejeitados = eventos.filter(F.col("motivo_rejeicao").isNotNull())
validos = eventos.filter(F.col("motivo_rejeicao").isNull()).drop("motivo_rejeicao")

por_usuario = Window.partitionBy("user_pseudo_id").orderBy("event_ts", "event_id")
usuario_todo = Window.partitionBy("user_pseudo_id").orderBy("event_ts", "event_id").rowsBetween(
    Window.unboundedPreceding, Window.unboundedFollowing)

silver = (validos
          .withColumn("_anterior", F.lag("event_ts").over(por_usuario))
          .withColumn("_nova", F.when(
              F.col("_anterior").isNull()
              | ((F.unix_timestamp("event_ts") - F.unix_timestamp("_anterior")) > SESSAO_SEGUNDOS), 1).otherwise(0))
          .withColumn("_numero", F.sum("_nova").over(por_usuario))
          .withColumn("session_id", F.concat_ws("-", "user_pseudo_id", F.col("_numero").cast("string")))
          .withColumn("user_id", F.first("user_id", ignorenulls=True).over(usuario_todo))
          .drop("_anterior", "_nova", "_numero"))
silver.cache()

itens = (silver.select("event_id", "event_name", "event_ts", "user_pseudo_id", "user_id", "session_id",
                       "transaction_id", "year", "month", "day", F.explode("items").alias("item"))
         .select("*", "item.*").drop("item"))

gravar_eventos(silver, "silver", "overwrite")
itens.write.mode("overwrite").partitionBy("year", "month", "day").parquet(caminho("silver/itens"))
rejeitados.write.mode("overwrite").parquet(caminho("silver/eventos_rejeitados"))

titulo("Silver")
print(f"eventos na bronze: {total_bronze}")
print(f"duplicatas descartadas: {total_bronze - total_dedup}")
print(f"eventos válidos: {silver.count()}")
print(f"itens: {itens.count()}")
rejeitados.groupBy("motivo_rejeicao").count().orderBy(F.desc("count")).show(truncate=False)
spark.stop()
