"""Earth (seed 0): where each wild species starts, and where the ores are.

HOMELANDS  species key -> [(lat, lon, radius_km), ...]: the wild ancestor's core range or
           domestication centre. Ranges then spread from these through suitable habitat
           as on any world (nature.species_native), up to REACH_KM.
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
    # wild einkorn, emmer and barley: the Fertile Crescent
    "temperate_grain": [(37.0, 40.0, 450), (34.0, 46.5, 350), (32.5, 36.0, 250)],
    # rye and oats: Anatolia and the Caucasus, spreading north as weeds of wheat
    "cold_grain": [(39.5, 42.0, 450), (44.0, 40.0, 400)],
    # wild rice: the middle and lower Yangtze; the Ganges plain
    "wet_grain": [(30.0, 116.0, 400), (30.5, 112.0, 300), (25.5, 85.0, 400)],
    # sorghum and pearl millet: the Sahel and Sudan; foxtail and broomcorn millet in north China
    "tropical_grain": [(13.0, 30.0, 500), (14.0, -2.0, 600), (36.0, 111.0, 400)],
    # teosinte, maize's ancestor: the Balsas valley
    "slow_grain": [(18.3, -100.5, 250)],
    # lentils, peas, chickpeas (Fertile Crescent); soybean (north China); beans (Mesoamerica, Andes)
    "legume": [(37.0, 40.0, 450), (36.0, 116.0, 400), (19.0, -99.0, 350), (-12.0, -76.0, 300)],
    # wild potato: around Lake Titicaca and the central Andes
    "highland_tuber": [(-15.8, -69.5, 350), (-12.0, -75.0, 300)],
    # yams (West Africa), taro and yams (New Guinea, Southeast Asia), cassava (southern Amazonia)
    "tropical_tuber": [(7.5, 2.0, 500), (-5.5, 143.0, 450), (16.0, 102.0, 400), (-11.0, -60.0, 500)],
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
# real range is smaller than the habitat model would allow.
REACH_KM = {
    "temperate_grain": 1200, "cold_grain": 1500, "slow_grain": 700, "highland_tuber": 900,
    "sheep_goat": 1800, "horse": 2500, "camel": 1800, "llama": 1400, "fowl": 1500,
}

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
