import json
from datetime import datetime, timezone

from comum import arquivo_do_projeto, caminho, criar_sessao, exigir_camada, titulo

spark = criar_sessao("exportar_mapa_calor")
exigir_camada(spark, "gold/mapa_calor_cliques", "python pipeline/gold_mapa_calor.py")
exigir_camada(spark, "gold/mapa_calor_rolagem", "python pipeline/gold_mapa_calor.py")

cliques = [linha.asDict() for linha in spark.read.parquet(caminho("gold/mapa_calor_cliques"))
           .orderBy("pagina", "element_id").collect()]
rolagem = [linha.asDict() for linha in spark.read.parquet(caminho("gold/mapa_calor_rolagem"))
           .orderBy("marco").collect()]
dados = {
    "gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cliques": cliques,
    "rolagem": rolagem,
}

destino = arquivo_do_projeto("site/mapa_calor.js")
destino.write_text("window.MAPA_CALOR = " + json.dumps(dados, ensure_ascii=False, default=float) + ";\n",
                   encoding="utf-8")

titulo("Exportação do mapa de calor")
print(f"{destino}: {len(cliques)} cliques, {len(rolagem)} marcos")
spark.stop()
