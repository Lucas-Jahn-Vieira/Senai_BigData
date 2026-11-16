from functools import reduce

from pyspark.sql import functions as F

from _gold_comum import gravar, iniciar, sessoes
from comum import titulo

MARCOS_VIDEO = [10, 25, 50, 75, 100]

spark, silver = iniciar("gold_engajamento")
dim = sessoes(silver)

tabela_sessoes = (silver.groupBy("session_id").agg(
    F.count("*").alias("eventos"),
    F.max("percent_scrolled").alias("scroll_maximo"),
    F.max("video_percent").alias("video_percent_maximo"),
    F.max(F.col("event_name") == "generate_lead").alias("fez_lead"),
    F.max(F.col("event_name") == "purchase").alias("comprou"))
    .join(dim, "session_id")
    .withColumn("duracao_segundos", F.unix_timestamp("fim") - F.unix_timestamp("inicio"))
    .select("session_id", "user_pseudo_id", "user_id", "inicio", "fim", "duracao_segundos", "eventos",
            "scroll_maximo", "video_percent_maximo", "fez_lead", "comprou", "canal", "dispositivo", "plano"))

video = (silver.filter(F.col("event_name").startswith("video_"))
         .groupBy("session_id").agg(
             F.max(F.col("event_name") == "video_start").alias("deu_play"),
             F.max("video_percent").alias("maximo"))
         .filter("deu_play"))
retencao_video = reduce(lambda a, b: a.unionByName(b), [
    video.agg(
        F.lit(m).alias("marco"),
        F.count("*").alias("sessoes_com_play"),
        F.sum(F.when(F.col("maximo") >= m, 1).otherwise(0)).alias("sessoes_que_atingiram"))
    for m in MARCOS_VIDEO
]).withColumn("percentual", F.try_divide("sessoes_que_atingiram", "sessoes_com_play"))

buscas_faq = (silver.filter(F.col("event_name") == "site_search")
              .withColumn("termo", F.lower(F.trim("search_term")))
              .groupBy("termo").agg(
                  F.count("*").alias("total_buscas"),
                  F.sum(F.when(F.col("search_results") == 0, 1).otherwise(0)).alias("buscas_sem_resultado")))

saidas = (silver.filter((F.col("event_name") == "click") & F.col("outbound"))
          .groupBy(F.col("link_domain").alias("item")).count()
          .withColumn("tipo", F.lit("saida")))
downloads = (silver.filter(F.col("event_name") == "file_download")
             .groupBy(F.col("file_name").alias("item")).count()
             .withColumn("tipo", F.lit("download")))
saidas_downloads = saidas.unionByName(downloads).withColumnRenamed("count", "total").select("tipo", "item", "total")

gravar(tabela_sessoes, "sessoes")
gravar(retencao_video, "retencao_video")
gravar(buscas_faq, "buscas_faq")
gravar(saidas_downloads, "saidas_downloads")

titulo("Gold engajamento")
retencao_video.orderBy("marco").show(truncate=False)
buscas_faq.orderBy(F.desc("total_buscas")).show(truncate=False)
saidas_downloads.orderBy("tipo", F.desc("total")).show(truncate=False)
spark.stop()
