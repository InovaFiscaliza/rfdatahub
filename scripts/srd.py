from extracao.datasources.srd import SRD

if __name__ == "__main__":
    import time

    start = time.perf_counter()

    data = SRD()

    # data.update()

    print("DATA")

    print(data.extraction.info())

    print(150 * "=")

    # print("DISCARDED!")

    # print(data.discarded[["Frequência", "Entidade", "Log"]])

    # print(150 * "=")

    # print(data.df.Multiplicidade.sum())

    # data.save()

    print(f"Elapsed time: {time.perf_counter() - start} seconds")