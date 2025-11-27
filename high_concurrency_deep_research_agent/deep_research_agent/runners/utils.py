
def read_queries_from_file(path: str) -> list:
        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(line)
        return queries