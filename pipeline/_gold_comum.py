from pyspark.sql import functions as F

from comum import caminho, criar_sessao, exigir_camada, glob_eventos, ler_eventos


def iniciar(nome):
    spark = criar_sessao(nome)
    exigir_camada(spark, glob_eventos("silver"), "python pipeline/silver.py")
    return spark, ler_eventos(spark, "silver")


def gravar(df, nome):
    df.write.mode("overwrite").parquet(caminho(f"gold/{nome}"))


def pessoa():
    return F.coalesce("user_id", "user_pseudo_id")


def pagina(coluna="page_location"):
    trecho = F.regexp_extract(F.col(coluna), r"^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]*([^?#]*)", 1)
    return F.when(F.col(coluna).isNull(), None).when(trecho == "", "/").otherwise(trecho)


def ordem():
    return F.struct("event_ts", "event_id")


def sessoes(silver):
    canal = F.concat_ws(" / ", F.coalesce("traffic_source", F.lit("(nenhum)")),
                        F.coalesce("traffic_medium", F.lit("(nenhum)")))
    base = silver.groupBy("session_id").agg(
        F.first("user_pseudo_id").alias("user_pseudo_id"),
        F.max("user_id").alias("user_id"),
        F.min_by(canal, ordem()).alias("canal"),
        F.min_by("device_category", ordem()).alias("dispositivo"),
        F.min("event_ts").alias("inicio"),
        F.max("event_ts").alias("fim"),
    )
    planos = (silver.filter(F.col("event_name") == "select_item")
              .groupBy("session_id")
              .agg(F.min_by(F.get("items", 0)["item_id"], ordem()).alias("plano")))
    return base.join(planos, "session_id", "left").fillna({"plano": "(nenhum)"})


def compras(silver, dim_sessoes):
    plano = F.get(F.filter("items", lambda i: i["item_category"] == "plano"), 0)["item_id"]
    return (silver.filter(F.col("event_name") == "purchase")
            .select("transaction_id", "session_id", "event_ts",
                    pessoa().alias("pessoa"),
                    F.to_date("event_ts").alias("data_compra"),
                    F.col("value").alias("receita"),
                    F.coalesce(plano, F.lit("(nenhum)")).alias("plano"),
                    F.exists("items", lambda i: i["item_id"] == "BUMP-MENTORIA").alias("tem_bump"))
            .join(dim_sessoes.select("session_id", "canal"), "session_id", "left"))


def reembolsos(silver):
    return (silver.filter(F.col("event_name") == "refund")
            .groupBy("transaction_id").agg(F.sum("value").alias("reembolso")))
