COMPOSITION = {
    "Kistik veya tamamen kistik": 0,
    "Süngerimsi": 0,
    "Mikst kistik-solid": 1,
    "Solid veya hemen tamamen solid": 2,
}
ECHOGENICITY = {
    "Anekoik": 0,
    "Hiperekoik veya izoekoik": 1,
    "Hipoekoik": 2,
    "Çok hipoekoik": 3,
}
SHAPE = {"Genişliği yüksekliğinden fazla": 0, "Yüksekliği genişliğinden fazla": 3}
MARGIN = {
    "Düzgün": 0,
    "Belirsiz": 0,
    "Lobüle veya düzensiz": 2,
    "Ekstratiroidal uzanım": 3,
}
FOCI = {
    "Yok veya büyük kuyruklu artefakt": 0,
    "Makrokalsifikasyon": 1,
    "Periferik/rim kalsifikasyon": 2,
    "Punktat ekojenik odaklar": 3,
}

def calculate_tirads(composition: str, echogenicity: str, shape: str, margin: str, foci: list[str]) -> dict:
    score = COMPOSITION[composition] + ECHOGENICITY[echogenicity] + SHAPE[shape] + MARGIN[margin]
    score += sum(FOCI[item] for item in foci)
    if score == 0:
        category = "TR1"
    elif score == 2:
        category = "TR2"
    elif score == 3:
        category = "TR3"
    elif 4 <= score <= 6:
        category = "TR4"
    else:
        category = "TR5"
    return {"score": score, "category": category}
