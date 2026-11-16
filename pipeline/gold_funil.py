from functools import reduce

from pyspark.sql import functions as F

from _gold_comum import gravar, iniciar, pagina, sessoes
from comum import titulo

ETAPAS = ["view_item_list", "select_item", "add_to_cart", "view_cart", "begin_checkout",
          "add_payment_info", "purchase"]
TODOS = "(todos)"

spark, silver = iniciar("gold_funil")
dim = sessoes(silver)


def teve(condicao):
    return F.max(F.when(condicao, 1).otherwise(0))


flags = silver.groupBy("session_id").agg(
    *[teve(F.col("event_name") == e).alias(e) for e in ETAPAS + ["view_item"]])
base = flags.join(dim.select("session_id", "plano", "canal"), "session_id")
com_total = base.unionByName(base.withColumn("plano", F.lit(TODOS)).withColumn("canal", F.lit(TODOS)))
contagens = com_total.groupBy("plano", "canal").agg(
    *[F.sum(e).alias(e) for e in ETAPAS + ["view_item"]])

linhas = [
    contagens.select(
        "plano", "canal",
        F.lit(i + 1).alias("ordem"),
        F.lit(etapa).alias("etapa"),
        F.col(etapa).alias("sessoes"),
        (F.try_divide(F.col(etapa), F.col(ETAPAS[i - 1])) if i else F.lit(None).cast("double"))
        .alias("conversao_etapa_anterior"),
        F.try_divide(F.col(etapa), F.col(ETAPAS[0])).alias("conversao_topo"),
        F.try_divide(F.col("view_item"), F.col("select_item")).alias("taxa_view_item"),
    )
    for i, etapa in enumerate(ETAPAS)
]
funil_compra = reduce(lambda a, b: a.unionByName(b), linhas)

na_landing = pagina() == "/"
lead = (silver.groupBy("session_id").agg(
    teve((F.col("event_name") == "page_view") & na_landing).alias("page_view_landing"),
    teve((F.col("event_name") == "scroll") & na_landing & (F.col("percent_scrolled") >= 50)).alias("scroll_50"),
    teve(F.col("event_name") == "generate_lead").alias("generate_lead"))
    .join(dim.select("session_id", "canal"), "session_id"))
lead = lead.unionByName(lead.withColumn("canal", F.lit(TODOS)))
funil_lead = (lead.groupBy("canal").agg(
    F.sum("page_view_landing").alias("sessoes_page_view_landing"),
    F.sum("scroll_50").alias("sessoes_scroll_50"),
    F.sum("generate_lead").alias("sessoes_generate_lead"))
    .withColumn("conversao_scroll", F.try_divide("sessoes_scroll_50", "sessoes_page_view_landing"))
    .withColumn("conversao_lead", F.try_divide("sessoes_generate_lead", "sessoes_scroll_50"))
    .withColumn("conversao_lead_topo", F.try_divide("sessoes_generate_lead", "sessoes_page_view_landing")))

gravar(funil_compra, "funil_compra")
gravar(funil_lead, "funil_lead")

titulo("Gold funil")
funil_compra.filter(F.col("plano") == TODOS).orderBy("ordem").show(truncate=False)
funil_lead.orderBy("canal").show(truncate=False)
spark.stop()
