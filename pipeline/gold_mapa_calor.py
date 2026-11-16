from functools import reduce

from pyspark.sql import Window
from pyspark.sql import functions as F

from _gold_comum import gravar, iniciar, pagina
from comum import titulo

MARCOS = [0, 25, 50, 75, 90]

spark, silver = iniciar("gold_mapa_calor")
eventos = silver.withColumn("pagina", pagina())

na_pagina = (eventos.filter((F.col("event_name") == "page_view") & F.col("pagina").isNotNull())
             .groupBy("pagina").agg(F.countDistinct("session_id").alias("sessoes_na_pagina")))

mapa_calor_cliques = (eventos.filter(F.col("element_id").isNotNull() & F.col("pagina").isNotNull())
                      .groupBy("pagina", "element_id").agg(
                          F.count("*").alias("cliques"),
                          F.countDistinct("session_id").alias("sessoes"))
                      .join(na_pagina, "pagina", "left")
                      .withColumn("taxa_clique",
                                  F.round(F.try_divide(F.col("sessoes"), F.col("sessoes_na_pagina")).cast("double"), 4))
                      .withColumn("intensidade", F.round(
                          (F.col("cliques") / F.max("cliques").over(Window.partitionBy("pagina"))).cast("double"), 4))
                      .select("pagina", "element_id", "cliques", "sessoes", "sessoes_na_pagina",
                              "taxa_clique", "intensidade"))

landing = (eventos.filter((F.col("event_name") == "page_view") & (F.col("pagina") == "/"))
           .select("session_id").distinct())
rolagem = (eventos.filter((F.col("event_name") == "scroll") & (F.col("pagina") == "/"))
           .groupBy("session_id").agg(F.max("percent_scrolled").alias("maximo")))
base = landing.join(rolagem, "session_id", "left").withColumn("maximo", F.coalesce("maximo", F.lit(0)))
total = base.count()

mapa_calor_rolagem = reduce(lambda a, b: a.unionByName(b), [
    base.agg(F.sum(F.when(F.col("maximo") >= m, 1).otherwise(0)).cast("long").alias("sessoes_que_chegaram"))
    .select(F.lit("/").alias("pagina"), F.lit(m).alias("marco"), "sessoes_que_chegaram")
    for m in MARCOS
]).withColumn("pct_sessoes", F.round(F.try_divide(F.col("sessoes_que_chegaram"), F.lit(total)).cast("double"), 4))

gravar(mapa_calor_cliques, "mapa_calor_cliques")
gravar(mapa_calor_rolagem, "mapa_calor_rolagem")

titulo("Gold mapa de calor")
mapa_calor_cliques.orderBy("pagina", F.desc("cliques")).show(50, truncate=False)
mapa_calor_rolagem.orderBy("marco").show(truncate=False)
spark.stop()
