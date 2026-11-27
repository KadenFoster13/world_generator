"""Earth (seed 0): where each wild species starts, where soils differ, and where the ores are.

HOMELANDS  species key -> [(lat, lon, radius_km), ...]: the wild ancestor's core range or
           domestication centre. Ranges then spread from these through suitable habitat
           as on any world (nature.species_native), up to REACH_KM.
SOILS      [(lat, lon, radius_km, fertility 0..1), ...]: soil regions the biome alone
           misjudges (old shields, leached tropical uplands, black earths, loess, volcanic).
ORES       mineral -> [(lat, lon, radius_km, richness 0..1), ...]: known ore regions,
           mostly those worked in antiquity or famous since. Elsewhere a damped copy of
           the generator's geology applies (iron stays mostly geological).

Positions are approximate (the map's cells are ~25-100 km); they stand for regions, not
mines, and follow the general consensus of archaeology and economic geology rather than
one source. For further reading: Zohary, Hopf & Weiss, "Domestication of Plants in the
Old World"; Larson et al. (2014, PNAS) on animal domestication; the USGS's mineral
resource publications for ore regions. Corrections welcome.
"""

# -------------------------------------------------------------- wild species
HOMELANDS = {
    # wild einkorn, emmer and barley: the Fertile Crescent; foxtail and broomcorn millet: the
    # middle Yellow River (a temperate dry-country grain like wheat, tamed there ~8000 BCE)
    "temperate_grain": [(37.0, 40.0, 450), (34.0, 46.5, 350), (32.5, 36.0, 250), (35.0, 111.0, 400)],
    # rye and oats: Anatolia and the Caucasus, spreading north as weeds of wheat
    "cold_grain": [(39.5, 42.0, 450), (44.0, 40.0, 400)],
    # wild rice: the middle and lower Yangtze; the Ganges plain
    "wet_grain": [(30.0, 116.0, 400), (30.5, 112.0, 300), (25.5, 85.0, 400)],
    # sorghum and pearl millet: the Sahel and Sudan; browntop and little millet: the Deccan
    "tropical_grain": [(13.0, 30.0, 500), (14.0, -2.0, 600), (16.0, 77.0, 400)],
    # teosinte, maize's ancestor: the Balsas valley
    "slow_grain": [(18.3, -100.5, 250)],
    # lentils, peas, chickpeas (Fertile Crescent); soybean (north China); beans (Mesoamerica, Andes)
    "legume": [(37.0, 40.0, 450), (36.0, 116.0, 400), (19.0, -99.0, 350), (-12.0, -76.0, 300)],
    # wild potato: around Lake Titicaca and the central Andes
    "highland_tuber": [(-15.8, -69.5, 350), (-12.0, -75.0, 300)],
    # yams (West Africa), taro and yams (New Guinea), cassava (southern Amazonia); mainland
    # Southeast Asia farmed rice first (taro there came later, as a minor crop)
    "tropical_tuber": [(7.5, 2.0, 500), (-5.5, 143.0, 450), (-11.0, -60.0, 500)],
    # olive and grape (Levant, Aegean), date palm (Gulf), banana (New Guinea), cacao (upper Amazon)
    "tree_crop": [(33.0, 35.5, 300), (38.0, 24.0, 300), (30.0, 47.5, 300), (-5.5, 143.0, 400),
                  (-3.0, -74.0, 350)],
    # cotton (Indus, Mesoamerica, coastal Peru), flax (Fertile Crescent), hemp (north China)
    "fibre": [(27.0, 68.5, 350), (18.0, -96.0, 350), (-8.0, -79.0, 250), (37.0, 40.0, 350), (35.0, 110.0, 400)],
    # aurochs: across the Near East and Europe; zebu: the Indus; and North Africa
    "cattle": [(37.0, 40.0, 500), (48.0, 15.0, 600), (27.0, 68.5, 400), (32.0, 5.0, 400)],
    # mouflon and bezoar goat: the Zagros and eastern Anatolia
    "sheep_goat": [(35.5, 46.0, 400), (38.5, 41.0, 350)],
    # wild boar: Anatolia, central China, Europe
    "pig": [(38.0, 40.0, 400), (33.0, 110.0, 500), (48.0, 15.0, 500)],
    # wild horse: the Pontic-Caspian steppe and Kazakhstan
    "horse": [(48.5, 44.0, 500), (51.0, 66.0, 600)],
    # dromedary (southern Arabia) and Bactrian camel (Mongolia, Gobi)
    "camel": [(18.0, 48.0, 500), (44.0, 100.0, 500)],
    # guanaco and vicuna: the Andean altiplano
    "llama": [(-17.0, -68.5, 400), (-10.0, -76.0, 300)],
    # red junglefowl: Southeast Asia and south China
    "fowl": [(19.0, 100.0, 500), (22.0, 108.0, 400)],
}

# How far each species' wild range may spread from its homelands on Earth (km): where the
# real range is smaller than the habitat model would allow. (Wild emmer, einkorn and barley
# grow within ~500-700 km of the Fertile Crescent: the Levant, SE Anatolia, the Zagros.
# Rice was tamed on the Yangtze and reached mainland Southeast Asia only ~2500-2000 BCE.
# Teosinte grows only in southern and central Mexico; wild lentils, peas, soybeans and
# beans stay near their homelands: eastern North America had none to tame.)
REACH_KM = {
    "temperate_grain": 700, "cold_grain": 1500, "wet_grain": 600, "slow_grain": 400, "highland_tuber": 900,
    "legume": 600,
    "sheep_goat": 1800, "horse": 2500, "camel": 1800, "llama": 1400, "fowl": 1500,
}

# -------------------------------------------------------------------- soils
# Earth's soils where they differ most from what the biome alone suggests: [(lat, lon,
# radius_km, fertility 0..1)], blended in toward the centre (before river silt, which keeps
# valleys and deltas rich). After the FAO-UNESCO Soil Map of the World and the USDA soil
# orders: weathered oxisols and ultisols on old shields and tropical uplands are poor;
# vertisols, mollisols, loess and young volcanic soils are rich.
SOILS = [
    # poor: leached tropical uplands and old shields
    (17.5, 102.5, 550, 0.30),   # Indochina uplands and the Khorat plateau (sandy, leached)
    (20.5, 97.0, 350, 0.35),    # Shan and Burmese hills
    (1.0, 114.0, 650, 0.25),    # Borneo
    (-13.0, -50.0, 1000, 0.30), # Brazilian shield and the Cerrado (oxisols)
    (4.0, -60.0, 600, 0.25),    # Guiana shield
    (-4.0, -63.0, 1300, 0.25),  # Amazon terra firme
    (-10.0, 27.0, 1100, 0.35),  # miombo woodland belt
    (0.0, 22.0, 900, 0.25),     # Congo basin
    (8.0, -2.0, 700, 0.40),     # Guinea coast uplands
    (55.0, -80.0, 1300, 0.20),  # Canadian shield
    (63.0, 18.0, 650, 0.25),    # Fennoscandian shield
    (32.5, -83.5, 600, 0.45),   # southeastern US coastal plain (ultisols)
    # rich: black earths, loess, young volcanic soils
    (19.0, 76.0, 550, 0.80),    # Deccan black cotton soils (vertisols on the traps)
    (36.0, 112.0, 650, 0.85),   # loess plateau and North China plain
    (26.5, 82.0, 650, 0.80),    # Indo-Gangetic alluvium
    (9.0, 39.0, 450, 0.75),     # Ethiopian highlands (volcanic)
    (19.3, -99.0, 400, 0.85),   # Trans-Mexican volcanic belt
    (14.5, -90.5, 300, 0.80),   # Guatemalan and Salvadoran volcanic highlands
    (-7.3, 110.0, 400, 0.85),   # Java (volcanic)
]

# -------------------------------------------------------------------- ores
ORES = {
    "tin": [(50.3, -5.2, 120, 1.0),     # Cornwall
            (50.5, 13.0, 120, 0.9),     # Erzgebirge
            (42.5, -7.5, 200, 0.8),     # Galicia and northern Portugal
            (3.5, 101.5, 350, 1.0),     # Malay peninsula
            (25.0, 103.5, 250, 0.9),    # Yunnan (Gejiu)
            (-19.0, -66.0, 300, 0.9),   # Bolivian tin belt
            (9.5, 8.8, 150, 0.8),       # Jos plateau
            (34.0, 66.0, 250, 0.6),     # Afghanistan
            (-1.5, 106.0, 250, 0.8)],   # Bangka and Belitung
    "copper": [(35.0, 33.0, 120, 1.0),  # Cyprus
               (37.7, -6.6, 150, 0.9),  # Rio Tinto, Iberian pyrite belt
               (38.5, 39.5, 250, 0.8),  # eastern Anatolia (Ergani)
               (44.0, 22.0, 250, 0.8),  # Balkans (Rudna Glava, Bor)
               (30.6, 35.4, 120, 0.7),  # Wadi Arabah (Timna, Faynan)
               (24.0, 57.0, 200, 0.7),  # Oman (Magan)
               (-23.0, -69.0, 500, 1.0),  # Atacama
               (-12.0, 27.5, 400, 1.0),   # Central African copperbelt
               (47.0, -88.5, 250, 0.9),   # Lake Superior native copper
               (53.0, 59.0, 300, 0.7),    # Urals
               (47.5, 91.0, 300, 0.6),    # Altai
               (31.0, 117.5, 250, 0.7)],  # lower Yangtze (Tonglushan)
    "iron": [(31.5, 106.0, 400, 0.6),   # Sichuan
             (47.0, 1.5, 400, 0.6),     # France
             (48.0, 33.5, 250, 0.8),    # Kryvyi Rih
             (-22.0, -44.0, 300, 0.9),  # Minas Gerais
             (-22.5, 118.0, 400, 0.9),  # Pilbara
             (47.5, -92.5, 250, 0.8),   # Mesabi range
             (21.5, 85.0, 300, 0.7),    # Odisha and Jharkhand
             (67.8, 20.2, 200, 0.8),    # Kiruna
             (10.0, 12.0, 500, 0.5)],   # West African bloomery iron
    "gold": [(21.0, 33.0, 500, 1.0),    # Nubia
             (39.5, 27.5, 200, 0.7),    # Lydia (Pactolus)
             (45.5, 23.0, 150, 0.8),    # Transylvania
             (42.5, -6.8, 150, 0.7),    # Las Medulas
             (-26.2, 28.0, 250, 1.0),   # Witwatersrand
             (12.0, -10.0, 400, 0.8),   # Bambuk and Bure (West Africa)
             (6.0, -1.5, 250, 0.8),     # Gold Coast
             (-7.0, -76.0, 400, 0.7),   # Andes
             (39.0, -120.5, 300, 0.8),  # California
             (62.0, 130.0, 600, 0.7),   # Siberia
             (-31.0, 121.5, 300, 0.8)], # Western Australia
    "silver": [(37.7, 24.05, 80, 1.0),  # Laurion
               (37.6, -1.0, 150, 0.8),  # Iberia (Cartagena)
               (-19.6, -65.75, 150, 1.0),  # Potosi
               (22.8, -102.6, 250, 1.0),   # Zacatecas
               (21.0, -101.3, 200, 0.8),   # Guanajuato
               (50.6, 13.0, 120, 0.8),     # Erzgebirge
               (35.1, 132.4, 100, 0.8),    # Iwami
               (40.5, 30.0, 250, 0.6)],    # Anatolia
    "coal": [(51.5, -1.5, 300, 0.9), (51.5, 7.0, 150, 1.0), (50.5, 19.0, 150, 0.8), (48.0, 38.0, 200, 0.9),
             (38.0, -81.0, 400, 1.0), (54.0, 87.0, 300, 0.9), (38.0, 112.0, 400, 1.0), (23.5, 86.5, 200, 0.8),
             (-32.5, 151.0, 300, 0.8), (-26.0, 29.0, 300, 0.8)],
    "oil": [(27.0, 50.0, 500, 1.0),     # the Gulf
            (40.4, 50.0, 150, 0.8),     # Baku
            (31.5, -102.0, 300, 0.9),   # Permian basin
            (29.0, -92.0, 400, 0.8),    # Gulf of Mexico coast
            (8.0, -67.0, 400, 0.9),     # Venezuela
            (5.0, 6.0, 300, 0.8),       # Niger delta
            (61.0, 73.0, 500, 0.9),     # western Siberia
            (57.0, 3.0, 300, 0.7),      # North Sea
            (28.0, 18.0, 400, 0.7),     # Libya
            (56.0, -112.0, 300, 0.7)],  # Alberta
    "saltpeter": [(-22.0, -69.5, 300, 1.0),  # Atacama nitrate fields
                  (25.0, 86.0, 300, 0.7),    # Bihar
                  (34.0, 109.0, 300, 0.5)],  # north China
}


# ------------------------------------------------------------------ lakes
# Earth's elevation records lake beds, not lake surfaces, so the model can't tell lakes
# from low ground. Each lake: points along its length (one for a round lake), its surface
# (m), its area (km2; the Aral and Chad as they were before 20th-century irrigation and
# drought), and whether it has no outlet (its rivers end there). Big enough to show on the map; the Caspian is kept as
# an inland sea (it lies below sea level).
LAKES = [
    ("Superior", [(47.0, -91.5), (47.7, -88.5), (47.3, -85.0)], 183, 82100, False),
    ("Michigan", [(45.8, -86.8), (43.5, -87.2), (41.8, -87.0)], 176, 58000, False),
    ("Huron", [(45.9, -83.6), (44.5, -82.2), (43.3, -82.0)], 176, 59600, False),
    ("Erie", [(42.0, -83.2), (42.3, -81.0), (42.7, -79.2)], 174, 25700, False),
    ("Ontario", [(43.3, -79.6), (43.6, -77.5), (43.9, -76.4)], 75, 18960, False),
    ("Winnipeg", [(50.6, -96.9), (52.5, -97.8), (53.7, -98.5)], 217, 24500, False),
    ("Great Bear", [(66.0, -123.5), (66.0, -120.0)], 156, 31000, False),
    ("Great Slave", [(61.3, -116.5), (61.7, -113.5), (62.5, -110.8)], 156, 27200, False),
    ("Athabasca", [(58.6, -111.0), (59.3, -108.0)], 213, 7850, False),
    ("Nicaragua", [(12.2, -86.0), (11.1, -85.0)], 32, 8264, False),
    ("Titicaca", [(-15.4, -69.7), (-16.3, -69.0)], 3812, 8372, False),
    ("Great Salt Lake", [(41.5, -112.6), (40.8, -112.3)], 1280, 4400, True),
    ("Ladoga", [(61.6, 31.0), (60.6, 31.5)], 5, 17700, False),
    ("Onega", [(62.6, 35.0), (61.2, 35.5)], 33, 9700, False),
    ("Vanern", [(59.2, 12.7), (58.8, 14.0)], 44, 5650, False),
    ("Baikal", [(55.7, 109.5), (53.5, 108.0), (51.6, 104.5)], 456, 31500, False),
    ("Balkhash", [(46.3, 73.6), (46.6, 76.0), (46.6, 79.0)], 342, 16400, True),
    ("Aral", [(46.5, 61.0), (44.5, 59.5), (43.7, 58.8)], 53, 68000, True),
    ("Issyk-Kul", [(42.5, 76.3), (42.3, 78.0)], 1607, 6236, True),
    ("Van", [(38.6, 42.9)], 1640, 3755, True),
    ("Urmia", [(38.2, 45.4), (37.2, 45.6)], 1270, 5200, True),
    ("Victoria", [(-1.0, 33.0)], 1134, 68800, False),
    ("Tanganyika", [(-3.4, 29.3), (-6.0, 29.7), (-8.7, 31.0)], 773, 32600, False),
    ("Malawi", [(-9.6, 34.0), (-12.0, 34.5), (-14.3, 35.2)], 474, 29600, False),
    ("Turkana", [(4.5, 36.0), (2.6, 36.6)], 360, 6405, True),
    ("Chad", [(13.0, 14.5)], 282, 20000, True),
]

# Gorges too narrow for the stored grid (a few km wide in ~25 km cells): without them the
# basins behind them fill into great false lakes. Each: points along the river (lat, lon,
# river level m), cut into the ground on every map size.
GORGES = {
    "Danube, Iron Gates": [(44.82, 20.45, 70), (44.75, 21.3, 66), (44.65, 22.0, 58), (44.62, 22.6, 45),
                           (44.3, 23.0, 38), (43.9, 23.6, 28), (43.7, 24.5, 22)],
    "Yangtze, Three Gorges": [(29.56, 106.55, 160), (29.7, 107.4, 140), (30.8, 108.4, 110), (31.0, 109.5, 90),
                              (31.07, 109.9, 80), (31.04, 110.3, 70), (30.7, 111.3, 45), (30.33, 112.2, 30)],
    # From Mbandaka through the narrow channel across the Bateke plateau to Malebo Pool,
    # then down the falls (without the channel the river spills west through Gabon).
    "Congo, Livingstone Falls": [(0.05, 18.26, 300), (-1.0, 17.15, 296), (-2.16, 16.23, 292), (-3.18, 16.2, 286),
                                 (-3.85, 15.95, 280), (-4.3, 15.3, 270),(-4.9, 14.4, 200), (-5.5, 13.8, 100), (-5.8, 13.45, 20),
                                 (-5.85, 13.05, 5), (-6.0, 12.4, 0)],
    # The Niger's bend past Tombouctou and through the Tossaye gorge (without it the river
    # spills west into the Senegal).
    "Niger, Tombouctou bend": [(13.44, -6.26, 285), (14.5, -4.2, 268), (15.9, -3.5, 263), (16.77, -3.0, 260),
                               (16.95, -0.6, 255), (16.27, -0.04, 250), (15.4, 0.9, 240), (13.5, 2.1, 180)],
    # The Amur through the Lesser Khingan and down its lower course to the Sea of Okhotsk
    # (without it the river spills south down the Ussuri to the Sea of Japan).
    "Amur, lower course": [(50.27, 127.53, 125), (49.2, 129.7, 95), (48.8, 130.6, 80), (47.9, 132.6, 50),
                           (48.48, 135.07, 35), (49.6, 136.6, 25), (50.55, 137.0, 20), (51.4, 138.0, 12),
                           (52.4, 139.7, 6), (53.14, 140.7, 2), (52.95, 141.2, 0)],
    # The Euphrates (Karasu and Murat) through its Taurus canyons and down through Syria,
    # and the Tigris past Mosul (without them the Euphrates is a trickle in Syria and the
    # Tigris turns south down the Khabur).
    "Euphrates, Karasu": [(39.7, 39.5, 1150), (39.26, 38.49, 850), (38.80, 38.75, 690)],
    "Euphrates, Murat": [(38.74, 41.5, 1250), (38.69, 39.93, 850), (38.80, 38.75, 690)],
    "Euphrates, Taurus to Syria": [(38.80, 38.75, 690), (38.23, 39.12, 560), (37.48, 38.32, 380),
                                   (37.03, 37.98, 340), (36.83, 38.01, 325), (35.95, 39.0, 245),
                                   (35.33, 40.14, 200), (34.45, 40.92, 175), (34.13, 42.38, 110),
                                   (33.64, 42.83, 55), (33.42, 43.3, 45)],
    "Tigris": [(37.91, 40.24, 580), (37.71, 41.41, 470), (37.33, 42.19, 370), (36.34, 43.13, 220),
               (34.6, 43.68, 105), (34.2, 43.87, 65), (33.33, 44.4, 34), (32.5, 45.83, 18),
               (31.84, 47.15, 8), (31.0, 47.43, 3)],
    "St Lawrence": [(44.2, -76.5, 74), (45.0, -74.7, 50), (45.5, -73.5, 10), (46.35, -72.5, 5),
                    (46.8, -71.2, 2), (47.5, -70.0, 0)],
}
