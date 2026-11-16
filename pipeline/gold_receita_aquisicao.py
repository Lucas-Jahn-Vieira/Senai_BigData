from pyspark.sql import Window
from pyspark.sql import functions as F

from _gold_comum import compras, gravar, iniciar, pessoa, reembolsos, sessoes
from comum import titulo

TOTAL = "(total)"

spark, silver = iniciar("gold_receita_aquisicao")
dim = sessoes(silver)
pedidos = (compras(silver, dim)
           .join(reembolsos(silver), "transaction_id", "left")
           .withColumn("reembolso", F.coalesce("reembolso", F.lit(0.0)))
           .withColumn("teve_reembolso", F.col("reembolso") > 0))

receita_diaria = (pedidos.groupBy("data_compra", "plano", "canal").agg(
    F.round(F.sum("receita"), 2).alias("receita_bruta"),
    F.round(F.sum("reembolso"), 2).alias("receita_reembolsada"),
    F.count("*").alias("pedidos"))
    .withColumn("receita_liquida", F.round(F.col("receita_bruta") - F.col("receita_reembolsada"), 2))
    .withColumn("ticket_medio", F.round(F.try_divide("receita_bruta", "pedidos"), 2)))

taxa_reembolso = (pedidos.groupBy("plano", "canal").agg(
    F.count("*").alias("pedidos"),
    F.sum(F.col("teve_reembolso").cast("int")).alias("pedidos_com_reembolso"))
    .withColumn("taxa_reembolso", F.try_divide("pedidos_com_reembolso", "pedidos")))

adesao_order_bump = (pedidos.agg(
    F.count("*").alias("pedidos"),
    F.sum(F.col("tem_bump").cast("int")).alias("pedidos_com_bump"))
    .withColumn("percentual_bump", F.try_divide("pedidos_com_bump", "pedidos")))

primeiro_lead = (silver.filter(F.col("event_name") == "generate_lead")
                 .withColumn("pessoa", pessoa())
                 .withColumn("_n", F.row_number().over(Window.partitionBy("pessoa").orderBy("event_ts", "event_id")))
                 .filter("_n = 1")
                 .select("pessoa", F.col("event_ts").alias("ts_lead"), "session_id")
                 .join(dim.select("session_id", "canal"), "session_id", "left"))

primeira_compra = (primeiro_lead.join(pedidos.select("pessoa", F.col("event_ts").alias("ts_compra")), "pessoa")
                   .filter(F.col("ts_compra") > F.col("ts_lead"))
                   .groupBy("pessoa").agg(F.min("ts_compra").alias("ts_compra")))

leads = (primeiro_lead.join(primeira_compra, "pessoa", "left")
         .withColumn("dias", (F.unix_micros("ts_compra") - F.unix_micros("ts_lead")) / 1e6 / 86400))
leads = leads.unionByName(leads.withColumn("canal", F.lit(TOTAL)))
lead_para_compra = (leads.groupBy("canal").agg(
    F.count("*").alias("leads"),
    F.count("ts_compra").alias("compraram_depois"),
    F.percentile_approx("dias", 0.5).alias("mediana_dias"))
    .withColumn("taxa_conversao", F.try_divide("compraram_depois", "leads")))

gravar(receita_diaria, "receita_diaria")
gravar(taxa_reembolso, "taxa_reembolso")
gravar(adesao_order_bump, "adesao_order_bump")
gravar(lead_para_compra, "lead_para_compra")

titulo("Gold receita e aquisição")
receita_diaria.agg(F.sum("receita_bruta"), F.sum("receita_reembolsada"), F.sum("receita_liquida"),
                   F.sum("pedidos")).show(truncate=False)
taxa_reembolso.orderBy("plano", "canal").show(truncate=False)
adesao_order_bump.show(truncate=False)
lead_para_compra.orderBy("canal").show(truncate=False)
spark.stop()
