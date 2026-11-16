from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from _gold_comum import gravar, iniciar, pessoa, sessoes
from comum import titulo

CATEGORICAS = ["canal", "dispositivo", "plano_selecionado"]
NUMERICAS = ["scroll_maximo", "video_percent_maximo", "viu_planos", "selecionou_plano", "abriu_detalhe",
             "buscas", "downloads", "cliques"]
SEGMENTO = ["sessoes", "eventos", "scroll_medio", "video_percent_maximo", "leads", "compras"]
GRUPOS = 4
SEMENTE = 42

spark, silver = iniciar("gold_ml")
dim = sessoes(silver)


def conta(nome):
    return F.sum(F.when(F.col("event_name") == nome, 1).otherwise(0))


def teve(nome):
    return F.max(F.when(F.col("event_name") == nome, 1).otherwise(0))


primeiro_carrinho = (silver.filter(F.col("event_name") == "add_to_cart")
                     .groupBy("session_id").agg(F.min("event_ts").alias("ts_carrinho")))
antes = (silver.join(primeiro_carrinho, "session_id", "left")
         .filter(F.col("ts_carrinho").isNull() | (F.col("event_ts") < F.col("ts_carrinho"))))

features = (antes.groupBy("session_id").agg(
    F.coalesce(F.max("percent_scrolled"), F.lit(0)).cast("double").alias("scroll_maximo"),
    F.coalesce(F.max("video_percent"), F.lit(0)).cast("double").alias("video_percent_maximo"),
    teve("view_item_list").cast("double").alias("viu_planos"),
    teve("select_item").cast("double").alias("selecionou_plano"),
    teve("view_item").cast("double").alias("abriu_detalhe"),
    conta("site_search").cast("double").alias("buscas"),
    conta("file_download").cast("double").alias("downloads"),
    conta("click").cast("double").alias("cliques"),
    F.min_by(F.when(F.col("event_name") == "select_item", F.get("items", 0)["item_id"]),
             F.when(F.col("event_name") == "select_item", F.struct("event_ts", "event_id"))).alias("plano_selecionado")))

rotulos = silver.groupBy("session_id").agg(teve("purchase").cast("double").alias("label"))

dados = (dim.select("session_id", "canal", "dispositivo")
         .join(rotulos, "session_id")
         .join(features, "session_id", "left")
         .fillna(0.0, NUMERICAS)
         .fillna("(nenhum)", CATEGORICAS))

positivos = dados.filter("label = 1").count()
total = dados.count()
peso_positivo = (total - positivos) / max(positivos, 1)
dados = dados.withColumn("peso", F.when(F.col("label") == 1, F.lit(peso_positivo)).otherwise(F.lit(1.0)))

treino, teste = dados.randomSplit([0.8, 0.2], seed=SEMENTE)

pipeline = Pipeline(stages=[
    StringIndexer(inputCols=CATEGORICAS, outputCols=[f"{c}_idx" for c in CATEGORICAS], handleInvalid="keep"),
    OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAS], outputCols=[f"{c}_vec" for c in CATEGORICAS],
                  handleInvalid="keep"),
    VectorAssembler(inputCols=NUMERICAS + [f"{c}_vec" for c in CATEGORICAS], outputCol="features"),
    LogisticRegression(featuresCol="features", labelCol="label", weightCol="peso", maxIter=50),
])
modelo = pipeline.fit(treino)

previsto_teste = modelo.transform(teste)
auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(previsto_teste)
matriz = previsto_teste.agg(
    F.sum(F.when((F.col("prediction") == 1) & (F.col("label") == 1), 1).otherwise(0)).alias("vp"),
    F.sum(F.when((F.col("prediction") == 1) & (F.col("label") == 0), 1).otherwise(0)).alias("fp"),
    F.sum(F.when((F.col("prediction") == 0) & (F.col("label") == 1), 1).otherwise(0)).alias("fn"),
).first()
precisao = matriz.vp / (matriz.vp + matriz.fp) if matriz.vp + matriz.fp else 0.0
recall = matriz.vp / (matriz.vp + matriz.fn) if matriz.vp + matriz.fn else 0.0
f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0

metricas_modelo = spark.createDataFrame(
    [(float(auc), float(f1), float(precisao), float(recall), treino.count(), teste.count(),
      NUMERICAS + CATEGORICAS)],
    "auc double, f1 double, precisao double, recall double, sessoes_treino long, sessoes_teste long, "
    "features array<string>")

ids_teste = teste.select("session_id").withColumn("conjunto", F.lit("teste"))
propensao_sessao = (modelo.transform(dados)
                    .join(ids_teste, "session_id", "left")
                    .select("session_id",
                            vector_to_array("probability")[1].alias("probabilidade_compra"),
                            F.col("prediction").cast("boolean").alias("previsao_compra"),
                            F.col("label").cast("boolean").alias("comprou"),
                            F.coalesce("conjunto", F.lit("treino")).alias("conjunto")))

visitantes = (silver.withColumn("visitante", pessoa()).groupBy("visitante").agg(
    F.countDistinct("session_id").cast("double").alias("sessoes"),
    F.count("*").cast("double").alias("eventos"),
    F.coalesce(F.avg("percent_scrolled"), F.lit(0.0)).alias("scroll_medio"),
    F.coalesce(F.max("video_percent"), F.lit(0)).cast("double").alias("video_percent_maximo"),
    conta("generate_lead").cast("double").alias("leads"),
    conta("purchase").cast("double").alias("compras")))

segmentador = Pipeline(stages=[
    VectorAssembler(inputCols=SEGMENTO, outputCol="bruto"),
    StandardScaler(inputCol="bruto", outputCol="features", withMean=True, withStd=True),
    KMeans(featuresCol="features", predictionCol="grupo", k=GRUPOS, seed=SEMENTE),
]).fit(visitantes)
agrupados = segmentador.transform(visitantes).select("visitante", "grupo", *SEGMENTO)
perfil = agrupados.groupBy("grupo").agg(
    F.count("*").alias("visitantes_no_grupo"),
    *[F.round(F.avg(c), 4).alias(f"media_{c}") for c in SEGMENTO])
segmentos_visitantes = agrupados.join(perfil, "grupo")

gravar(propensao_sessao, "propensao_sessao")
gravar(metricas_modelo, "metricas_modelo")
gravar(segmentos_visitantes, "segmentos_visitantes")

titulo("Gold ML")
metricas_modelo.show(truncate=False)
perfil.orderBy("grupo").show(truncate=False)
spark.stop()
