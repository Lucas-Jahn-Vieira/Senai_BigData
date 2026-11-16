from pyspark.sql import functions as F
from pyspark.sql.types import (ArrayType, BooleanType, DoubleType, IntegerType, LongType, StringType,
                               StructField, StructType)

from comum import camada_existe, criar_sessao, exigir_camada, glob_eventos, gravar_eventos, ler_eventos, titulo

ITEM = StructType([
    StructField("item_id", StringType()),
    StructField("item_name", StringType()),
    StructField("item_category", StringType()),
    StructField("price", DoubleType()),
    StructField("quantity", LongType()),
])

PARAMS = StructType([
    StructField("page_title", StringType()),
    StructField("percent_scrolled", IntegerType()),
    StructField("link_url", StringType()),
    StructField("link_domain", StringType()),
    StructField("link_text", StringType()),
    StructField("link_id", StringType()),
    StructField("outbound", BooleanType()),
    StructField("search_term", StringType()),
    StructField("search_results", IntegerType()),
    StructField("search_area", StringType()),
    StructField("video_title", StringType()),
    StructField("video_provider", StringType()),
    StructField("video_duration", DoubleType()),
    StructField("video_current_time", DoubleType()),
    StructField("video_percent", IntegerType()),
    StructField("file_name", StringType()),
    StructField("file_extension", StringType()),
    StructField("item_list_id", StringType()),
    StructField("item_list_name", StringType()),
    StructField("currency", StringType()),
    StructField("value", DoubleType()),
    StructField("shipping_tier", StringType()),
    StructField("shipping", DoubleType()),
    StructField("payment_type", StringType()),
    StructField("transaction_id", StringType()),
    StructField("tax", DoubleType()),
    StructField("coupon", StringType()),
    StructField("reason", StringType()),
    StructField("lead_source", StringType()),
    StructField("method", StringType()),
    StructField("element_id", StringType()),
])

SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("event_name", StringType()),
    StructField("event_ts", StringType()),
    StructField("user_pseudo_id", StringType()),
    StructField("user_id", StringType()),
    StructField("session_hint", StringType()),
    StructField("page", StructType([
        StructField("location", StringType()),
        StructField("referrer", StringType()),
    ])),
    StructField("traffic", StructType([
        StructField("source", StringType()),
        StructField("medium", StringType()),
        StructField("campaign", StringType()),
    ])),
    StructField("device", StructType([
        StructField("category", StringType()),
        StructField("os", StringType()),
        StructField("browser", StringType()),
    ])),
    StructField("params", PARAMS),
    StructField("items", ArrayType(ITEM)),
])

spark = criar_sessao("bronze")
exigir_camada(spark, glob_eventos("landing"), "python gerador/gerar_eventos.py")

landing = (ler_eventos(spark, "landing", "json", schema=SCHEMA)
           .withColumn("arquivo_origem", F.col("_metadata.file_path"))
           .withColumn("data_ingestao", F.current_timestamp()))

if camada_existe(spark, glob_eventos("bronze")):
    ja_ingeridos = ler_eventos(spark, "bronze").select("arquivo_origem").distinct()
    landing = landing.join(ja_ingeridos, "arquivo_origem", "left_anti")

total = landing.count()
gravar_eventos(landing, "bronze", "append")

titulo("Bronze")
print(f"eventos ingeridos: {total}")
spark.stop()
