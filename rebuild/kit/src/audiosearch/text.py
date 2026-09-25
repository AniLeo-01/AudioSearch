import re

STOP = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    let me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves also yeah okay oh um uh
    like really got get gets going go know think thing things lot kind sort well right mean
    actually""".split()  # noqa: SIM905 - compact word list
)
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    return TOKEN.findall(re.sub(r"'s\b", "", text).replace("'", ""))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if len(t) >= 3 and t not in STOP and not t.isdigit()]
