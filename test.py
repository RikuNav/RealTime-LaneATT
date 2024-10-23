import laneatt.utils.anchors 
import os
import torch
import unittest
import yaml

import numpy as np

class TestAnchors(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAnchors, self).__init__(*args, **kwargs)

        config_file=os.path.join(os.path.dirname(__file__), 'laneatt', 'config', 'laneatt.yaml')
        self.__laneatt_config = yaml.safe_load(open(config_file))

        self.left_angles = self.__laneatt_config['anchor_angles']['left']
        self.right_angles = self.__laneatt_config['anchor_angles']['right']
        self.bottom_angles = self.__laneatt_config['anchor_angles']['bottom']
        self.y_discretization = self.__laneatt_config['anchor_discretization']['y']
        self.x_discretization = self.__laneatt_config['anchor_discretization']['x']
        self.img_size = (self.__laneatt_config['image_size']['height'], self.__laneatt_config['image_size']['width'])

    def almost_equal_lists(self, list1, list2):
        for i in range(len(list1)):
            if not np.isclose(list1[i], list2[i], atol=0.001):
                return False
        return True

    def test_left_anchor(self):
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0, 0.49295774647887325), 60, 72, (64, 10, 20), (720, 1280)), [ 0,          0,        365.0704,  0,         0,         -210.7735,
                                                                                                                                                    -204.9187, -199.0639, -193.2091, -187.3542, -181.4994, -175.6446,
                                                                                                                                                    -169.7898, -163.9350, -158.0801, -152.2253, -146.3705, -140.5157,
                                                                                                                                                    -134.6608, -128.8060, -122.9512, -117.0964, -111.2416, -105.3867,
                                                                                                                                                    -99.5320,  -93.6771,  -87.8223,  -81.9675,  -76.1127,  -70.2578,
                                                                                                                                                    -64.4030,  -58.5482,  -52.6934,  -46.8386,  -40.9837,  -35.1289,
                                                                                                                                                    -29.2741,  -23.4193,  -17.5645,  -11.7096,   -5.8548,    0.0000,
                                                                                                                                                    5.8548,   11.7096,   17.5645,   23.4193,   29.2741,   35.1289,
                                                                                                                                                    40.9837,   46.8386,   52.6934,   58.5482,   64.4030,   70.2578,
                                                                                                                                                    76.1127,   81.9675,   87.8223,   93.6771,   99.5319,  105.3868,
                                                                                                                                                    111.2416,  117.0964,  122.9512,  128.8060,  134.6609,  140.5157,
                                                                                                                                                    146.3705,  152.2253,  158.0801,  163.9350,  169.7898,  175.6446,
                                                                                                                                                    181.4994,  187.3542,  193.2090,  199.0639,  204.9187]))
    
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0, 1), 22, 72, (64, 10, 20), (720, 1280)), [   0.0000,    0.0000,    0.0000,    0.0000,    0.0000,    0.0000,
                                                                                                                                    25.0995,   50.1990,   75.2984,  100.3979,  125.4974,  150.5968,
                                                                                                                                    175.6963,  200.7957,  225.8952,  250.9947,  276.0942,  301.1937,
                                                                                                                                    326.2932,  351.3926,  376.4921,  401.5916,  426.6910,  451.7905,
                                                                                                                                    476.8899,  501.9894,  527.0889,  552.1884,  577.2878,  602.3873,
                                                                                                                                    627.4868,  652.5862,  677.6857,  702.7852,  727.8848,  752.9841,
                                                                                                                                    778.0836,  803.1831,  828.2825,  853.3821,  878.4814,  903.5810,
                                                                                                                                    928.6804,  953.7799,  978.8794, 1003.9789, 1029.0784, 1054.1777,
                                                                                                                                    1079.2772, 1104.3767, 1129.4762, 1154.5757, 1179.6750, 1204.7747,
                                                                                                                                    1229.8740, 1254.9736, 1280.0731, 1305.1725, 1330.2721, 1355.3715,
                                                                                                                                    1380.4708, 1405.5704, 1430.6698, 1455.7695, 1480.8689, 1505.9683,
                                                                                                                                    1531.0677, 1556.1672, 1581.2668, 1606.3662, 1631.4656, 1656.5651,
                                                                                                                                    1681.6647, 1706.7642, 1731.8635, 1756.9629, 1782.0625]))                                                 

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0, 0), 39, 72, (64, 10, 20), (720, 1280)), [   0.0000,    0.0000,  720.0000,    0.0000,    0.0000, -889.1259,
                                                                                                                                    -876.6030, -864.0801, -851.5573, -839.0344, -826.5114, -813.9885,
                                                                                                                                    -801.4656, -788.9428, -776.4199, -763.8969, -751.3740, -738.8511,
                                                                                                                                    -726.3282, -713.8054, -701.2824, -688.7595, -676.2366, -663.7137,
                                                                                                                                    -651.1909, -638.6679, -626.1450, -613.6221, -601.0992, -588.5763,
                                                                                                                                    -576.0534, -563.5305, -551.0076, -538.4847, -525.9618, -513.4389,
                                                                                                                                    -500.9160, -488.3931, -475.8702, -463.3473, -450.8244, -438.3015,
                                                                                                                                    -425.7786, -413.2557, -400.7328, -388.2099, -375.6870, -363.1641,
                                                                                                                                    -350.6412, -338.1183, -325.5954, -313.0725, -300.5496, -288.0267,
                                                                                                                                    -275.5038, -262.9809, -250.4580, -237.9351, -225.4122, -212.8893,
                                                                                                                                    -200.3664, -187.8435, -175.3206, -162.7977, -150.2748, -137.7519,
                                                                                                                                    -125.2290, -112.7061, -100.1832,  -87.6603,  -75.1374,  -62.6145,
                                                                                                                                    -50.0916,  -37.5687,  -25.0458,  -12.5229,    0.0000]))

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0, 0.49295774647887325), 60, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000,  5.0704,  0.0000,  0.0000, -2.9274, -2.2859, -1.6444,
                                                                                                                                                                -1.0029, -0.3614,  0.2801,  0.9216,  1.5631,  2.2046,  2.8461]))
        
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0, 1), 22, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  2.7501,  5.5002,
                                                                                                                                            8.2503, 11.0004, 13.7505, 16.5006, 19.2507, 22.0008, 24.7509]))
        
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0, 0), 39, 72, (64, 10, 20), (720, 1280), fv=True), [  0.0000,   0.0000,  10.0000,   0.0000,   0.0000, -12.3490, -10.9769,
                                                                                                                                    -9.6048,  -8.2326,  -6.8605,  -5.4884,  -4.1163,  -2.7442,  -1.3721,
                                                                                                                                    0.0000]))

    def test_right_anchor(self):
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((1, 0.49295774647887325), 120, 72, (64, 10, 20), (720, 1280)), [   0.0000,    0.0000,  365.0704, 1280.0000,    0.0000, 1490.7736,
                                                                                                                                                        1484.9187, 1479.0638, 1473.2091, 1467.3542, 1461.4994, 1455.6445,
                                                                                                                                                        1449.7898, 1443.9349, 1438.0801, 1432.2253, 1426.3705, 1420.5156,
                                                                                                                                                        1414.6609, 1408.8060, 1402.9512, 1397.0964, 1391.2416, 1385.3867,
                                                                                                                                                        1379.5320, 1373.6771, 1367.8223, 1361.9675, 1356.1127, 1350.2578,
                                                                                                                                                        1344.4031, 1338.5482, 1332.6934, 1326.8385, 1320.9838, 1315.1289,
                                                                                                                                                        1309.2742, 1303.4193, 1297.5645, 1291.7096, 1285.8549, 1280.0000,
                                                                                                                                                        1274.1451, 1268.2904, 1262.4355, 1256.5807, 1250.7260, 1244.8711,
                                                                                                                                                        1239.0162, 1233.1615, 1227.3066, 1221.4518, 1215.5969, 1209.7422,
                                                                                                                                                        1203.8873, 1198.0325, 1192.1777, 1186.3229, 1180.4680, 1174.6133,
                                                                                                                                                        1168.7584, 1162.9036, 1157.0488, 1151.1940, 1145.3391, 1139.4844,
                                                                                                                                                        1133.6295, 1127.7747, 1121.9199, 1116.0651, 1110.2102, 1104.3555,
                                                                                                                                                        1098.5006, 1092.6458, 1086.7910, 1080.9362, 1075.0813]))
    
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((1, 1), 158, 72, (64, 10, 20), (720, 1280)), [     0.0000,      0.0000,      0.0000,   1280.0000,      0.0000,
                                                                                                                                        1280.0000,   1254.9005,   1229.8010,   1204.7017,   1179.6022,
                                                                                                                                        1154.5027,   1129.4032,   1104.3037,   1079.2043,   1054.1047,
                                                                                                                                        1029.0052,   1003.9058,    978.8063,    953.7068,    928.6074,
                                                                                                                                        903.5079,    878.4084,    853.3090,    828.2095,    803.1101,
                                                                                                                                        778.0106,    752.9111,    727.8116,    702.7122,    677.6127,
                                                                                                                                        652.5132,    627.4138,    602.3143,    577.2148,    552.1152,
                                                                                                                                        527.0159,    501.9164,    476.8169,    451.7175,    426.6179,
                                                                                                                                        401.5186,    376.4190,    351.3196,    326.2201,    301.1206,
                                                                                                                                        276.0211,    250.9216,    225.8223,    200.7228,    175.6233,
                                                                                                                                        150.5238,    125.4243,    100.3250,     75.2253,     50.1260,
                                                                                                                                            25.0264,     -0.0731,    -25.1725,    -50.2721,    -75.3715,
                                                                                                                                        -100.4708,   -125.5704,   -150.6698,   -175.7695,   -200.8689,
                                                                                                                                        -225.9683,   -251.0677,   -276.1672,   -301.2668,   -326.3662,
                                                                                                                                        -351.4656,   -376.5651,   -401.6647,   -426.7642,   -451.8635,
                                                                                                                                        -476.9629,   -502.0625]))                                                 

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((1, 0), 141, 72, (64, 10, 20), (720, 1280)), [   0.0000,    0.0000,  720.0000, 1280.0000,    0.0000, 2169.1260,
                                                                                                                                    2156.6030, 2144.0801, 2131.5574, 2119.0344, 2106.5115, 2093.9885,
                                                                                                                                    2081.4656, 2068.9429, 2056.4199, 2043.8970, 2031.3740, 2018.8511,
                                                                                                                                    2006.3281, 1993.8054, 1981.2825, 1968.7595, 1956.2366, 1943.7136,
                                                                                                                                    1931.1909, 1918.6680, 1906.1450, 1893.6221, 1881.0992, 1868.5763,
                                                                                                                                    1856.0535, 1843.5305, 1831.0076, 1818.4847, 1805.9618, 1793.4390,
                                                                                                                                    1780.9160, 1768.3931, 1755.8702, 1743.3473, 1730.8245, 1718.3015,
                                                                                                                                    1705.7786, 1693.2557, 1680.7328, 1668.2100, 1655.6870, 1643.1641,
                                                                                                                                    1630.6412, 1618.1183, 1605.5955, 1593.0725, 1580.5496, 1568.0267,
                                                                                                                                    1555.5038, 1542.9810, 1530.4580, 1517.9351, 1505.4122, 1492.8893,
                                                                                                                                    1480.3665, 1467.8435, 1455.3206, 1442.7977, 1430.2748, 1417.7520,
                                                                                                                                    1405.2290, 1392.7061, 1380.1832, 1367.6603, 1355.1375, 1342.6145,
                                                                                                                                    1330.0916, 1317.5687, 1305.0458, 1292.5229, 1280.0000]))

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((1, 0.49295774647887325), 120, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000,  5.0704, 20.0000,  0.0000, 22.9274, 22.2859, 21.6444,
                                                                                                                                                                21.0029, 20.3614, 19.7199, 19.0784, 18.4369, 17.7954, 17.1539]))
        
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((1, 1), 158, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000,  0.0000, 20.0000,  0.0000, 20.0000, 17.2499, 14.4998,
                                                                                                                                            11.7497,  8.9996,  6.2495,  3.4994,  0.7493, -2.0008, -4.7509]))
        
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((1, 0), 141, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000, 10.0000, 20.0000,  0.0000, 32.3490, 30.9769, 29.6048,
                                                                                                                                            28.2326, 26.8605, 25.4884, 24.1163, 22.7442, 21.3721, 20.0000]))

    def test_bottom_anchor(self):
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0.5, 1), 100, 72, (64, 10, 20), (720, 1280)), [  0.0000,   0.0000,   0.0000, 640.0000,   0.0000, 640.0000, 638.2119,
                                                                                                                                        636.4238, 634.6357, 632.8476, 631.0594, 629.2714, 627.4833, 625.6952,
                                                                                                                                        623.9070, 622.1190, 620.3309, 618.5427, 616.7546, 614.9666, 613.1784,
                                                                                                                                        611.3903, 609.6022, 607.8141, 606.0260, 604.2379, 602.4498, 600.6617,
                                                                                                                                        598.8736, 597.0855, 595.2974, 593.5093, 591.7212, 589.9330, 588.1450,
                                                                                                                                        586.3569, 584.5688, 582.7806, 580.9926, 579.2045, 577.4163, 575.6282,
                                                                                                                                        573.8401, 572.0520, 570.2639, 568.4758, 566.6877, 564.8996, 563.1115,
                                                                                                                                        561.3234, 559.5353, 557.7472, 555.9591, 554.1710, 552.3829, 550.5948,
                                                                                                                                        548.8066, 547.0186, 545.2305, 543.4424, 541.6542, 539.8661, 538.0781,
                                                                                                                                        536.2899, 534.5018, 532.7137, 530.9256, 529.1375, 527.3494, 525.5613,
                                                                                                                                        523.7732, 521.9851, 520.1970, 518.4089, 516.6208, 514.8327, 513.0446]))
        
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0.25, 1), 30, 72, (64, 10, 20), (720, 1280)), [   0.0000,    0.0000,    0.0000,  320.0000,    0.0000,  320.0000,
                                                                                                                                        337.5645,  355.1289,  372.6934,  390.2578,  407.8223,  425.3868,
                                                                                                                                        442.9512,  460.5156,  478.0801,  495.6446,  513.2090,  530.7736,
                                                                                                                                        548.3380,  565.9024,  583.4669,  601.0314,  618.5958,  636.1603,
                                                                                                                                        653.7247,  671.2892,  688.8536,  706.4181,  723.9825,  741.5471,
                                                                                                                                        759.1115,  776.6759,  794.2404,  811.8049,  829.3694,  846.9338,
                                                                                                                                        864.4982,  882.0627,  899.6271,  917.1917,  934.7560,  952.3206,
                                                                                                                                        969.8849,  987.4495, 1005.0139, 1022.5784, 1040.1428, 1057.7073,
                                                                                                                                        1075.2717, 1092.8362, 1110.4006, 1127.9651, 1145.5295, 1163.0941,
                                                                                                                                        1180.6584, 1198.2229, 1215.7874, 1233.3518, 1250.9165, 1268.4808,
                                                                                                                                        1286.0452, 1303.6097, 1321.1741, 1338.7388, 1356.3031, 1373.8676,
                                                                                                                                        1391.4320, 1408.9965, 1426.5610, 1444.1254, 1461.6898, 1479.2543,
                                                                                                                                        1496.8188, 1514.3833, 1531.9478, 1549.5121, 1567.0767]))
        
        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0.75, 1), 165, 72, (64, 10, 20), (720, 1280)), [    0.0000,     0.0000,     0.0000,   960.0000,     0.0000,   960.0000,
                                                                                                                                        922.1538,   884.3076,   846.4616,   808.6155,   770.7693,   732.9231,
                                                                                                                                        695.0769,   657.2309,   619.3847,   581.5385,   543.6924,   505.8462,
                                                                                                                                        468.0000,   430.1540,   392.3078,   354.4616,   316.6154,   278.7693,
                                                                                                                                        240.9232,   203.0770,   165.2309,   127.3848,    89.5386,    51.6924,
                                                                                                                                        13.8464,   -23.9998,   -61.8461,   -99.6921,  -137.5385,  -175.3844,
                                                                                                                                        -213.2306,  -251.0768,  -288.9230,  -326.7692,  -364.6151,  -402.4614,
                                                                                                                                        -440.3074,  -478.1537,  -515.9998,  -553.8461,  -591.6921,  -629.5382,
                                                                                                                                        -667.3844,  -705.2305,  -743.0768,  -780.9229,  -818.7689,  -856.6152,
                                                                                                                                        -894.4613,  -932.3075,  -970.1536, -1007.9996, -1045.8461, -1083.6921,
                                                                                                                                        -1121.5381, -1159.3843, -1197.2305, -1235.0769, -1272.9229, -1310.7688,
                                                                                                                                        -1348.6150, -1386.4614, -1424.3076, -1462.1536, -1499.9995, -1537.8459,
                                                                                                                                        -1575.6921, -1613.5383, -1651.3843, -1689.2302, -1727.0767]))

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0.5, 1), 100, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000,  0.0000, 10.0000,  0.0000, 10.0000,  9.8041,  9.6082,
                                                                                                                                                9.4122,  9.2163,  9.0204,  8.8245,  8.6286,  8.4326,  8.2367]))

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0.25, 1), 30, 72, (64, 10, 20), (720, 1280), fv=True), [ 0.0000,  0.0000,  0.0000,  5.0000,  0.0000,  5.0000,  6.9245,  8.8490,
                                                                                                                                                10.7735, 12.6980, 14.6225, 16.5470, 18.4715, 20.3960, 22.3205]))

        self.assertTrue(self.almost_equal_lists(laneatt.utils.anchors.generate_anchor((0.75, 1), 165, 72, (64, 10, 20), (720, 1280), fv=True), [  0.0000,   0.0000,   0.0000,  15.0000,   0.0000,  15.0000,  10.8533,
                                                                                                                                                6.7066,   2.5598,  -1.5869,  -5.7336,  -9.8803, -14.0271, -18.1738,
                                                                                                                                                -22.3205]))

    def test_left_anchors(self):
        anchors = laneatt.utils.anchors.generate_side_anchors(self.left_angles, 
                                                              self.y_discretization, 
                                                              (64, 10, 20), 
                                                              self.y_discretization, 
                                                              self.img_size, 
                                                              x=0.)
        ys = np.linspace(1, 0, self.y_discretization)
        angles_number = len(self.left_angles)
        repeated_ys = np.repeat(ys, angles_number)
        for i, anchor in enumerate(anchors[0]):
            self.assertTrue(self.almost_equal_lists(anchor, laneatt.utils.anchors.generate_anchor((0, repeated_ys[i]),
                                                                                                    self.left_angles[i%angles_number],
                                                                                                    self.y_discretization, 
                                                                                                    (64, 10, 20), 
                                                                                                    self.img_size)))
            
        for i, anchor in enumerate(anchors[1]):
            self.assertTrue(self.almost_equal_lists(anchor, laneatt.utils.anchors.generate_anchor((0, repeated_ys[i]),
                                                                                                    self.left_angles[i%angles_number],
                                                                                                    self.y_discretization, 
                                                                                                    (64, 10, 20), 
                                                                                                    self.img_size, fv=True)))
            
    def test_right_anchors(self):
        anchors = laneatt.utils.anchors.generate_side_anchors(self.right_angles, 
                                                              self.y_discretization, 
                                                              (64, 10, 20), 
                                                              self.y_discretization,
                                                              self.img_size,
                                                              x=1.)
        ys = np.linspace(1, 0, self.y_discretization)
        angles_number = len(self.right_angles)
        repeated_ys = np.repeat(ys, angles_number)
        for i, anchor in enumerate(anchors[0]):
            self.assertTrue(self.almost_equal_lists(anchor, laneatt.utils.anchors.generate_anchor((1, repeated_ys[i]),
                                                                                                    self.right_angles[i%angles_number],
                                                                                                    self.y_discretization, 
                                                                                                    (64, 10, 20), 
                                                                                                    self.img_size)))
            
        for i, anchor in enumerate(anchors[1]):
            self.assertTrue(self.almost_equal_lists(anchor, laneatt.utils.anchors.generate_anchor((1, repeated_ys[i]),
                                                                                                    self.right_angles[i%angles_number],
                                                                                                    self.y_discretization, 
                                                                                                    (64, 10, 20), 
                                                                                                    self.img_size, fv=True)))

    def test_bottom_anchors(self):
        anchors = laneatt.utils.anchors.generate_side_anchors(self.bottom_angles, 
                                                            self.x_discretization, 
                                                            (64, 10, 20), 
                                                            self.y_discretization, 
                                                            self.img_size, 
                                                            y=1.)
        xs = np.linspace(1, 0, self.x_discretization)
        angles_number = len(self.bottom_angles)
        repeated_xs = np.repeat(xs, angles_number)
        for i, anchor in enumerate(anchors[0]):
            self.assertTrue(self.almost_equal_lists(anchor, laneatt.utils.anchors.generate_anchor((repeated_xs[i], 1),
                                                                                                    self.bottom_angles[i%angles_number],
                                                                                                    self.y_discretization, 
                                                                                                    (64, 10, 20), 
                                                                                                    self.img_size)))
        for i, anchor in enumerate(anchors[1]):
            self.assertTrue(self.almost_equal_lists(anchor, laneatt.utils.anchors.generate_anchor((repeated_xs[i], 1),
                                                                                                    self.bottom_angles[i%angles_number],
                                                                                                    self.y_discretization, 
                                                                                                    (64, 10, 20), 
                                                                                                    self.img_size, fv=True)))

    def test_generate_anchors(self):
        anchors = laneatt.utils.anchors.generate_anchors(self.y_discretization, 
                                                        self.x_discretization, 
                                                        self.left_angles, 
                                                        self.right_angles, 
                                                        self.bottom_angles, 
                                                        (64, 10, 20), 
                                                        self.img_size)
        
        left_anchors = laneatt.utils.anchors.generate_side_anchors(self.left_angles, 
                                                                    self.y_discretization, 
                                                                    (64, 10, 20), 
                                                                    self.y_discretization, 
                                                                    self.img_size, 
                                                                    x=0.)
        
        right_anchors = laneatt.utils.anchors.generate_side_anchors(self.right_angles, 
                                                                    self.y_discretization, 
                                                                    (64, 10, 20), 
                                                                    self.y_discretization,
                                                                    self.img_size,
                                                                    x=1.)
        
        bottom_anchors = laneatt.utils.anchors.generate_side_anchors(self.bottom_angles, 
                                                            self.x_discretization, 
                                                            (64, 10, 20), 
                                                            self.y_discretization, 
                                                            self.img_size, 
                                                            y=1.)
        
        image_anchors = torch.cat([left_anchors[0], bottom_anchors[0], right_anchors[0]])
        feature_volume_anchors = torch.cat([left_anchors[1], bottom_anchors[1], right_anchors[1]]) 

        for anchor, image_anchor in zip(anchors[0], image_anchors):
            self.assertTrue(self.almost_equal_lists(anchor, image_anchor))

        for anchor, feature_volume_anchor in zip(anchors[1], feature_volume_anchors):
            self.assertTrue(self.almost_equal_lists(anchor, feature_volume_anchor))
            
if __name__ == '__main__':
    unittest.main()