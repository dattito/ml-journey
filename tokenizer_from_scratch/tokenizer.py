from collections import OrderedDict


def split_sub(t: list[str], sep: str) -> list[str]:
    res = []
    for e in t:
        n = e.split(sep)
        if len(n) == 1:
            res += n
        else:
            for f in n:
                if f != "":
                    res.append(f)
                res.append(sep)
            res.pop()

    return res


def count_pairs(original: list[list[int]]) -> list[tuple[tuple[int, int], int]]:
    pairs = {}
    for p in original:
        for i in range(len(p) - 1):
            t = p[i : i + 2]
            t = tuple(t)
            if pairs.get(t) is not None:
                pairs[t] += 1
            else:
                pairs[t] = 1

    pairs_sorted = list(pairs.items())
    pairs_sorted.sort(key=lambda x: x[1])
    return pairs_sorted


def replace_sublist(
    original: list[list[int]], target_sublist: list[int], replacement_value: int
):
    r = []
    for p in original:
        i = 0
        result_list = []
        if len(p) < len(target_sublist):
            r.append(p)
            continue
        while i < len(p):
            # Check if the current slice of the list matches the target_sublist
            # Ensure there are enough elements remaining for a potential match
            if p[i : i + len(target_sublist)] == target_sublist:
                # Match found: append the replacement value
                result_list.append(replacement_value)
                # Advance the index by the length of the target sublist
                # to effectively "skip" the matched elements
                i += len(target_sublist)
            else:
                # No match: append the current element
                result_list.append(p[i])
                # Advance the index by 1
                i += 1
        r.append(result_list)
    return r


def rereplace_sublist(
    original: list[list[int]], target_sublist: list[int], replacement_value: int
) -> list[list[int]]:
    r = []
    for p in original:
        result_list = []
        for e in p:
            if e == replacement_value:
                result_list += target_sublist
            else:
                result_list.append(e)
        r.append(result_list)
    return r


def decode_text(int_list: list[list[int]]) -> str:
    t = ""
    for p in int_list:
        byte_list = []
        for num in p:
            byte_list.append(num.to_bytes(1, byteorder="big", signed=False))
        full_bytes = b"".join(byte_list)
        t += full_bytes.decode("utf-8")
    return t


def decode_mapping(mappings: dict[int, tuple[tuple[int, int], int]], i) -> list[int]:
    decoded = mappings[i][0]
    result = []
    for e in decoded:
        if e < 256:
            result.append(e)
        else:
            result += decode_mapping(mappings, e)
    return result


class Tokenizer:
    def __init__(self, seps="0123456789 \n[]{}()\"'.,:;-_"):
        self.mappings = OrderedDict()
        self.seps = seps
        self.stats = []

    def fit(self, text: str, new_tokens: int) -> list[int]:
        content = [text]
        for e in self.seps:
            content = split_sub(content, e)

        content_bytes = list(map(lambda x: list(map(int, x.encode("utf-8"))), content))

        tokens = [i for i in range(256)]
        self.stats = [(sum(len(a) for a in content_bytes), len(tokens))]
        self.mappings = OrderedDict()

        for _ in range(new_tokens):
            pairs = count_pairs(content_bytes)
            new_token = len(tokens)
            tokens.append(new_token)
            to_be_replaced = pairs[-1][0]
            _count = pairs[-1][1]
            content_bytes = replace_sublist(
                content_bytes, list(to_be_replaced), new_token
            )
            self.stats.append((sum(len(a) for a in content_bytes), len(tokens)))
            self.mappings[new_token] = (to_be_replaced, _count)

        c = []
        for a in content_bytes:
            c += a
        return c

    def tokenize(self, text: str) -> list[int]:
        content = [text]
        for e in self.seps:
            content = split_sub(content, e)

        content_bytes = list(map(lambda x: list(map(int, x.encode("utf-8"))), content))

        tokens = [i for i in range(256)]
        self.stats = [(sum(len(a) for a in content_bytes), len(tokens))]

        for new_token, (to_be_replaced, _count) in self.mappings:
            content_bytes = replace_sublist(
                content_bytes, list(to_be_replaced), new_token
            )

        c = []
        for a in content_bytes:
            c += a
        return c

    def revert(self, tokens: list[int]) -> str:
        a = [tokens]
        for k, v in reversed(self.mappings.items()):
            a = rereplace_sublist(a, v[0], k)

        return decode_text(a)

    def print_mapping_counts(self):
        for k, v in self.mappings.items():
            print(f"'{decode_text([decode_mapping(self.mappings, k)])}' : {v[1]}")


if __name__ == "__main__":
    with open("Goethe--Faust.txt", "r") as f:
        content = f.read()

    t = Tokenizer()
    t.fit(content, 10)
    t.print_mapping_counts()
    print(t.stats)
