module kernel_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains

  subroutine columnar_multifluid_rgb_step(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t1422, t1426, t1430, t998, t995, t306, t315, t294, t297, t466, t475, t454, t457, t616, t625, t604, t607, t890, t900, t910, t961, t969, t1257, t35, t1243, t1244, t1245, t1246, t1247, t1248) bind(C, name="columnar_multifluid_rgb_step")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_1
    integer(c_int), intent(in), value :: extent_16
    real(c_double), intent(in) :: t0(extent_16)
    real(c_double), intent(in) :: t1(extent_16)
    real(c_double), intent(in) :: t2(extent_16)
    real(c_double), intent(in) :: t3(extent_16)
    real(c_double), intent(in) :: t4(extent_16)
    real(c_double), intent(in) :: t5(extent_16)
    real(c_double), intent(in) :: t6(extent_16)
    real(c_double), intent(in) :: t7(extent_16)
    real(c_double), intent(in) :: t8(extent_16)
    real(c_double), intent(in) :: t9(extent_16)
    real(c_double), intent(in) :: t10(extent_16)
    real(c_double), intent(in) :: t11(extent_16)
    real(c_double), intent(in) :: t12(extent_16)
    real(c_double), intent(in) :: t13(extent_16)
    real(c_double), intent(in) :: t14(extent_16)
    real(c_double), intent(in) :: t15(extent_16)
    real(c_double), intent(in) :: t16(extent_16)
    real(c_double), intent(in) :: t17(extent_16)
    real(c_double), intent(in) :: t18(extent_16)
    real(c_double), intent(in) :: t19(extent_16)
    real(c_double), intent(in) :: t20(extent_16)
    real(c_double), intent(in) :: t21(extent_16)
    real(c_double), intent(in) :: t22(extent_16)
    real(c_double), intent(in) :: t23(extent_16)
    real(c_double), intent(in) :: t24(extent_16)
    real(c_double), intent(in) :: t25(extent_16)
    real(c_double), intent(in) :: t26(extent_16)
    real(c_double), intent(in) :: t27(extent_16)
    real(c_double), intent(in) :: t28(extent_16)
    real(c_double), intent(in) :: t29(extent_16)
    real(c_double), intent(in) :: t30(extent_16)
    real(c_double), intent(in) :: t31(extent_16)
    real(c_double), intent(in) :: t32(extent_16)
    real(c_double), intent(in) :: t33(extent_16)
    real(c_double), intent(out) :: t1422(extent_16)
    real(c_double), intent(out) :: t1426(extent_16)
    real(c_double), intent(out) :: t1430(extent_16)
    real(c_double), intent(out) :: t998(extent_16)
    real(c_double), intent(out) :: t995(extent_16)
    real(c_double), intent(out) :: t306(extent_16)
    real(c_double), intent(out) :: t315(extent_16)
    real(c_double), intent(out) :: t294(extent_16)
    real(c_double), intent(out) :: t297(extent_16)
    real(c_double), intent(out) :: t466(extent_16)
    real(c_double), intent(out) :: t475(extent_16)
    real(c_double), intent(out) :: t454(extent_16)
    real(c_double), intent(out) :: t457(extent_16)
    real(c_double), intent(out) :: t616(extent_16)
    real(c_double), intent(out) :: t625(extent_16)
    real(c_double), intent(out) :: t604(extent_16)
    real(c_double), intent(out) :: t607(extent_16)
    real(c_double), intent(out) :: t890(extent_16)
    real(c_double), intent(out) :: t900(extent_16)
    real(c_double), intent(out) :: t910(extent_16)
    real(c_double), intent(out) :: t961(extent_16)
    real(c_double), intent(out) :: t969(extent_16)
    real(c_double), intent(out) :: t1257(extent_16)
    real(c_double), intent(out) :: t35(extent_16)
    real(c_double), intent(out) :: t1243(extent_16)
    real(c_double), intent(out) :: t1244(extent_16)
    real(c_double), intent(out) :: t1245(extent_16)
    real(c_double), intent(out) :: t1246(extent_16)
    real(c_double), intent(out) :: t1247(extent_16)
    real(c_double), intent(out) :: t1248(extent_16)
    real(c_double) :: t36(extent_16)
    real(c_double) :: t73(extent_16)
    real(c_double) :: t74(extent_16)
    real(c_double) :: t72(extent_16)
    real(c_double) :: t152(extent_16)
    real(c_double) :: t153(extent_16)
    real(c_double) :: t207(extent_16)
    real(c_double) :: t208(extent_16)
    real(c_double) :: t205(extent_16)
    real(c_double) :: t206(extent_16)
    real(c_double) :: t219(extent_16)
    real(c_double) :: t220(extent_16)
    real(c_double) :: t316(extent_16)
    real(c_double) :: t317(extent_16)
    real(c_double) :: t372(extent_16)
    real(c_double) :: t373(extent_16)
    real(c_double) :: t379(extent_16)
    real(c_double) :: t380(extent_16)
    real(c_double) :: t476(extent_16)
    real(c_double) :: t477(extent_16)
    real(c_double) :: t529(extent_16)
    real(c_double) :: t530(extent_16)
    real(c_double) :: t919(extent_16)
    real(c_double) :: t921(extent_16)
    real(c_double) :: t923(extent_16)
    real(c_double) :: t1312(extent_16)
    real(c_double) :: t1313(extent_16)
    real(c_double) :: t80(extent_16)
    real(c_double) :: t86(extent_16)
    real(c_double) :: t92(extent_16)
    real(c_double) :: t975(extent_16)
    real(c_double) :: t40(extent_16)
    real(c_double) :: t94(extent_16)
    real(c_double) :: t96(extent_16)
    real(c_double) :: t98(extent_16)
    real(c_double) :: t104(extent_16)
    real(c_double) :: t156(extent_16)
    real(c_double) :: t320(extent_16)
    real(c_double) :: t480(extent_16)
    real(c_double) :: t926(extent_16)
    real(c_double) :: t929(extent_16)
    real(c_double) :: t213(extent_16)
    real(c_double) :: t218(extent_16)
    real(c_double) :: t378(extent_16)
    real(c_double) :: t64(extent_16)
    real(c_double) :: t227(extent_16)
    real(c_double) :: t387(extent_16)
    real(c_double) :: t537(extent_16)
    real(c_double) :: t1136(extent_16)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t68(extent_16)
    real(c_double) :: t938(extent_16)
    real(c_double) :: t947(extent_16)
    real(c_double) :: t1146(extent_16)
    real(c_double) :: t172(extent_16)
    real(c_double) :: t336(extent_16)
    real(c_double) :: t496(extent_16)
    real(c_double) :: t1156(extent_16)
    real(c_double) :: t1325(extent_16)
    real(c_double) :: t1378(extent_16)
    real(c_double) :: t47(extent_16)
    real(c_double) :: t1387(extent_16)
    real(c_double) :: t1342(extent_16)
    real(c_double) :: t1389(extent_16)
    real(c_double) :: t1391(extent_16)
    real(c_double) :: t49(extent_16)
    real(c_double) :: t48(extent_16)
    real(c_double) :: t646(extent_16)
    real(c_double) :: t1355(extent_16)
    real(c_double) :: t1242(extent_16)
    real(c_double) :: t1357(extent_16)
    real(c_double) :: t71(extent_16)
    real(c_double) :: t132(extent_16)
    real(c_double) :: t151(extent_16)
    real(c_double) :: t359(extent_16)
    real(c_double) :: t516(extent_16)
    real(c_double) :: t192(extent_16)
    real(c_double) :: t653(extent_16)
    real(c_double) :: t660(extent_16)
    real(c_double) :: t667(extent_16)
    real(c_double) :: t772(extent_16)
    real(c_double) :: t763(extent_16)
    real(c_double) :: t781(extent_16)
    real(c_double) :: t722(extent_16)
    real(c_double) :: t874(extent_16)
    real(c_double) :: t877(extent_16)
    real(c_double) :: t880(extent_16)
    real(c_double) :: t754(extent_16)
    real(c_double) :: t1416(extent_16)
    real(c_double) :: t808(extent_1)
    real(c_double) :: t814(extent_1)
    real(c_double) :: t820(extent_1)
    real(c_double) :: t1181(extent_16)
    real(c_double) :: t1188(extent_16)
    real(c_double) :: t1195(extent_16)
    real(c_double) :: t1202(extent_16)
    real(c_double) :: t1209(extent_16)
    real(c_double) :: t1216(extent_16)
    real(c_double) :: t1418(extent_16)
    real(c_double) :: t811(extent_16)
    real(c_double) :: t817(extent_16)
    real(c_double) :: t823(extent_16)
    real(c_double) :: t1328(extent_16)
    real(c_double) :: t1265(extent_16)
    real(c_double) :: t1268(extent_16)
    real(c_double) :: t1285(extent_16)
    real(c_double) :: t1297(extent_16)
    real(c_double) :: t1299(extent_16)
    real(c_double) :: t1018(extent_16)
    real(c_double) :: t1009(extent_16)

    ! block entry
    t35 = (t0 + 0.025_c_double)
    t36 = (t1 - t2)
    t73 = (t8 + t9)
    t74 = (t10 + t11)
    t72 = (t6 + t7)
    t152 = (t4 - t16)
    t153 = (t5 - t17)
    t207 = (t16 - t20)
    t208 = (t17 - t21)
    t205 = (t16 - t18)
    t206 = (t17 - t19)
    t219 = (5.0_c_double - t16)
    t220 = (3.45_c_double - t17)
    t316 = (t4 - t18)
    t317 = (t5 - t19)
    t372 = (t18 - t20)
    t373 = (t19 - t21)
    t379 = (5.0_c_double - t18)
    t380 = (3.45_c_double - t19)
    t476 = (t4 - t20)
    t477 = (t5 - t21)
    t529 = (5.0_c_double - t20)
    t530 = (3.45_c_double - t21)
    t919 = (t4 - 1.15_c_double)
    t921 = (t5 - 5.75_c_double)
    t923 = (t4 - 8.85_c_double)
    t1312 = (t4 - 5.0_c_double)
    t1313 = (t5 - 3.45_c_double)
    t80 = min(max(t12, 0.0_c_double), 1.0_c_double)
    t86 = min(max(t13, 0.0_c_double), 1.0_c_double)
    t92 = min(max(t14, 0.0_c_double), 1.0_c_double)
    t975 = min(max(t30, 0.0_c_double), 0.42_c_double)
    t40 = (((t3 * 2.0_c_double) - t1) - t2)
    t94 = (1.0_c_double - t80)
    t96 = (1.0_c_double - t86)
    t98 = (1.0_c_double - t92)
    t104 = real(floor((t35 * 0.08_c_double)), c_double)
    t156 = ((t152 * t152) + (t153 * t153))
    t320 = ((t316 * t316) + (t317 * t317))
    t480 = ((t476 * t476) + (t477 * t477))
    t926 = ((t919 * t919) + (t921 * t921))
    t929 = ((t923 * t923) + (t921 * t921))
    t213 = (((t205 * t205) + (t206 * t206)) + 0.18_c_double)
    t218 = (((t207 * t207) + (t208 * t208)) + 0.18_c_double)
    t378 = (((t372 * t372) + (t373 * t373)) + 0.18_c_double)
    t64 = (((t4 * 0.61_c_double) + (t5 * 0.83_c_double)) + (sin(((t4 * 0.37_c_double) - (t5 * 0.29_c_double))) * 0.72_c_double))
    t227 = sqrt((((t219 * t219) + (t220 * t220)) + 0.08_c_double))
    t387 = sqrt((((t379 * t379) + (t380 * t380)) + 0.08_c_double))
    t537 = sqrt((((t529 * t529) + (t530 * t530)) + 0.08_c_double))
    t1136 = ((sin((t35 * 2.11_c_double)) * 0.22_c_double) + 0.78_c_double)
    t66 = cos(t64)
    t68 = sin(t64)
    t938 = exp(((-t926) / 0.4608_c_double))
    t947 = exp(((-t929) / 0.6728_c_double))
    t1146 = ((sin(((t35 * 1.91_c_double) + 2.1_c_double)) * 0.22_c_double) + 0.78_c_double)
    t172 = (exp(((-t156) / 4.805000000000001_c_double)) / (t156 + 0.14_c_double))
    t336 = (exp(((-t320) / 4.805000000000001_c_double)) / (t320 + 0.14_c_double))
    t496 = (exp(((-t480) / 4.805000000000001_c_double)) / (t480 + 0.14_c_double))
    t1156 = ((sin(((t35 * 2.27_c_double) + 4.2_c_double)) * 0.22_c_double) + 0.78_c_double)
    t1325 = exp(((-((t1312 * t1312) + (t1313 * t1313))) / 0.23120000000000004_c_double))
    t1378 = exp(((-t926) / 0.1058_c_double))
    t47 = sqrt((((t36 * t36) + (t40 * t40)) + 1.0e-05_c_double))
    t1387 = exp(((-t929) / 0.1058_c_double))
    t1342 = (1.0_c_double - t1325)
    t1389 = (1.0_c_double - t1378)
    t1391 = (1.0_c_double - t1387)
    t49 = (t40 / t47)
    t48 = (t36 / t47)
    t646 = ((min(max(((((t1 * 0.42_c_double) + (t3 * 0.34_c_double)) + (t2 * 0.18_c_double)) + (t28 * 0.35_c_double)), 0.0_c_double), 1.0_c_double) * 0.12_c_double) + 0.18_c_double)
    t1355 = min((t947 * t975), 0.6_c_double)
    t1242 = (1.0_c_double - min((t938 * 0.003_c_double), 0.08_c_double))
    t1357 = (1.0_c_double - t1355)
    t71 = ((t66 * t48) + (t68 * t49))
    t132 = (max(((sin((((t4 * 12.9898_c_double) + (t5 * 78.233_c_double)) + (t104 * 37.719_c_double))) * cos((((t4 * 39.3467_c_double) - (t5 * 11.135_c_double)) + (t104 * 19.913_c_double)))) - 0.72_c_double), 0.0_c_double) / 0.28_c_double)
    t151 = min(max(((t15 * 0.9999250028124297_c_double) + ((t132 * t132) * 0.0008_c_double)), 0.0_c_double), 1.0_c_double)
    t359 = ((t336 * (((((t96 * 1.2_c_double) * t72) - (t73 * 0.3_c_double)) + ((t96 * 1.65_c_double) * t151)) + (((t68 * t48) - (t66 * t49)) * 0.18_c_double))) / (sqrt((t320 + 0.08_c_double)) * (sum(t336) + 1.0e-06_c_double)))
    t516 = ((t496 * (((((t98 * 1.2_c_double) * t73) - (t74 * 0.3_c_double)) + ((t98 * 1.65_c_double) * t151)) - (t71 * 0.18_c_double))) / (sqrt((t480 + 0.08_c_double)) * (sum(t496) + 1.0e-06_c_double)))
    t192 = ((t172 * (((((t94 * 1.2_c_double) * t74) - (t72 * 0.3_c_double)) + ((t94 * 1.65_c_double) * t151)) + (t71 * 0.18_c_double))) / (sqrt((t156 + 0.08_c_double)) * (sum(t172) + 1.0e-06_c_double)))
    t294 = (t22 + ((((((((((t4 * 0.0_c_double) + sum((t152 * t192))) * 2.35_c_double) + (((t80 * 1.85_c_double) * t219) / t227)) + (cos(((t35 * 1.71_c_double) + 0.2_c_double)) * 0.58_c_double)) + ((t205 * 0.3_c_double) / t213)) + ((t207 * 0.3_c_double) / t218)) + ((5.0_c_double - t16) * 0.08_c_double)) - (t22 * 0.54_c_double)) * 0.025_c_double))
    t297 = (t23 + ((((((((((t5 * 0.0_c_double) + sum((t153 * t192))) * 2.35_c_double) + (((t80 * 1.85_c_double) * t220) / t227)) + (sin(((t35 * 1.37_c_double) + 1.1_c_double)) * 0.58_c_double)) + ((t206 * 0.3_c_double) / t213)) + ((t208 * 0.3_c_double) / t218)) + ((3.5_c_double - t17) * 0.08_c_double)) - (t23 * 0.54_c_double)) * 0.025_c_double))
    t454 = (t24 + ((((((((((t4 * 0.0_c_double) + sum((t316 * t359))) * 2.35_c_double) + (((t86 * 1.85_c_double) * t379) / t387)) + (cos(((t35 * 1.63_c_double) + 2.3_c_double)) * 0.58_c_double)) - ((t205 * 0.3_c_double) / t213)) + ((t372 * 0.3_c_double) / t378)) + ((5.0_c_double - t18) * 0.08_c_double)) - (t24 * 0.54_c_double)) * 0.025_c_double))
    t604 = (t26 + ((((((((((t4 * 0.0_c_double) + sum((t476 * t516))) * 2.35_c_double) + (((t92 * 1.85_c_double) * t529) / t537)) + (cos(((t35 * 1.79_c_double) + 4.2_c_double)) * 0.58_c_double)) - ((t207 * 0.3_c_double) / t218)) - ((t372 * 0.3_c_double) / t378)) + ((5.0_c_double - t20) * 0.08_c_double)) - (t26 * 0.54_c_double)) * 0.025_c_double))
    t457 = (t25 + ((((((((((t5 * 0.0_c_double) + sum((t317 * t359))) * 2.35_c_double) + (((t86 * 1.85_c_double) * t380) / t387)) + (sin(((t35 * 1.43_c_double) + 2.8_c_double)) * 0.58_c_double)) - ((t206 * 0.3_c_double) / t213)) + ((t373 * 0.3_c_double) / t378)) + ((3.5_c_double - t19) * 0.08_c_double)) - (t25 * 0.54_c_double)) * 0.025_c_double))
    t607 = (t27 + ((((((((((t5 * 0.0_c_double) + sum((t477 * t516))) * 2.35_c_double) + (((t92 * 1.85_c_double) * t530) / t537)) + (sin(((t35 * 1.31_c_double) + 5.0_c_double)) * 0.58_c_double)) - ((t208 * 0.3_c_double) / t218)) - ((t373 * 0.3_c_double) / t378)) + ((3.5_c_double - t21) * 0.08_c_double)) - (t27 * 0.54_c_double)) * 0.025_c_double))
    t306 = min(max((t16 + (t294 * 0.025_c_double)), 0.65_c_double), 9.35_c_double)
    t315 = min(max((t17 + (t297 * 0.025_c_double)), 0.65_c_double), 6.35_c_double)
    t466 = min(max((t18 + (t454 * 0.025_c_double)), 0.65_c_double), 9.35_c_double)
    t475 = min(max((t19 + (t457 * 0.025_c_double)), 0.65_c_double), 6.35_c_double)
    t616 = min(max((t20 + (t604 * 0.025_c_double)), 0.65_c_double), 9.35_c_double)
    t625 = min(max((t21 + (t607 * 0.025_c_double)), 0.65_c_double), 6.35_c_double)
    t653 = (((t4 - t306) * (t4 - t306)) + ((t5 - t315) * (t5 - t315)))
    t660 = (((t4 - t466) * (t4 - t466)) + ((t5 - t475) * (t5 - t475)))
    t667 = (((t4 - t616) * (t4 - t616)) + ((t5 - t625) * (t5 - t625)))
    t772 = exp(((-t660) / 0.18_c_double))
    t763 = exp(((-t653) / 0.18_c_double))
    t781 = exp(((-t667) / 0.18_c_double))
    t722 = max(max((min(max((t646 - abs((t4 - t306))), 0.0_c_double), max((t646 - abs((t5 - t315))), 0.0_c_double)) / t646), (min(max((t646 - abs((t4 - t466))), 0.0_c_double), max((t646 - abs((t5 - t475))), 0.0_c_double)) / t646)), (min(max((t646 - abs((t4 - t616))), 0.0_c_double), max((t646 - abs((t5 - t625))), 0.0_c_double)) / t646))
    t874 = ((t80 * exp(((-(((t306 - 5.0_c_double) * (t306 - 5.0_c_double)) + ((t315 - 3.45_c_double) * (t315 - 3.45_c_double)))) / 0.32000000000000006_c_double))) * 2.4_c_double)
    t877 = ((t86 * exp(((-(((t466 - 5.0_c_double) * (t466 - 5.0_c_double)) + ((t475 - 3.45_c_double) * (t475 - 3.45_c_double)))) / 0.32000000000000006_c_double))) * 2.4_c_double)
    t880 = ((t92 * exp(((-(((t616 - 5.0_c_double) * (t616 - 5.0_c_double)) + ((t625 - 3.45_c_double) * (t625 - 3.45_c_double)))) / 0.32000000000000006_c_double))) * 2.4_c_double)
    t754 = min(((exp(((-t653) / 1.2168_c_double)) + exp(((-t660) / 1.2168_c_double))) + exp(((-t667) / 1.2168_c_double))), 1.0_c_double)
    t1416 = (t722 * t722)
    t808 = min((sum((t151 * t763)) / (sum(t763) + 1.0e-06_c_double)), 0.7_c_double)
    t814 = min((sum((t151 * t772)) / (sum(t772) + 1.0e-06_c_double)), 0.7_c_double)
    t820 = min((sum((t151 * t781)) / (sum(t781) + 1.0e-06_c_double)), 0.7_c_double)
    t1181 = min(((t6 * 0.9976278180810777_c_double) + ((exp(((-t653) / 0.1152_c_double)) * 0.08000000000000002_c_double) * t1136)), 1.0_c_double)
    t1188 = min(((t7 * 0.9982016190284373_c_double) + ((exp(((-t653) / 0.2888_c_double)) * 0.05500000000000001_c_double) * t1136)), 1.0_c_double)
    t1195 = min(((t8 * 0.9976278180810777_c_double) + ((exp(((-t660) / 0.1152_c_double)) * 0.08000000000000002_c_double) * t1146)), 1.0_c_double)
    t1202 = min(((t9 * 0.9982016190284373_c_double) + ((exp(((-t660) / 0.2888_c_double)) * 0.05500000000000001_c_double) * t1146)), 1.0_c_double)
    t1209 = min(((t10 * 0.9976278180810777_c_double) + ((exp(((-t667) / 0.1152_c_double)) * 0.08000000000000002_c_double) * t1156)), 1.0_c_double)
    t1216 = min(((t11 * 0.9982016190284373_c_double) + ((exp(((-t667) / 0.2888_c_double)) * 0.05500000000000001_c_double) * t1156)), 1.0_c_double)
    t1418 = (1.0_c_double - t1416)
    t1243 = (t1181 * t1242)
    t1244 = (t1188 * t1242)
    t1246 = (t1202 * t1242)
    t1248 = (t1216 * t1242)
    t1245 = (t1195 * t1242)
    t1247 = (t1209 * t1242)
    t811 = ((t94 * t808(1)) * 1.35_c_double)
    t817 = ((t96 * t814(1)) * 1.35_c_double)
    t823 = ((t98 * t820(1)) * 1.35_c_double)
    t969 = max((t29 + (((t874 + t877) + t880) * 0.025_c_double)), 0.0_c_double)
    t1328 = min(t969, 1.0_c_double)
    t890 = min(max((t80 + ((t811 - t874) * 0.025_c_double)), 0.0_c_double), 1.0_c_double)
    t900 = min(max((t86 + ((t817 - t877) * 0.025_c_double)), 0.0_c_double), 1.0_c_double)
    t995 = (t32 + (((((((t754 * -0.42_c_double) - ((t722 * 0.22_c_double) * t722)) + ((t975 * 0.16_c_double) * t947)) - t31) * 20.0_c_double) - (t32 * 4.6_c_double)) * 0.025_c_double))
    t910 = min(max((t92 + ((t823 - t880) * 0.025_c_double)), 0.0_c_double), 1.0_c_double)
    t1265 = max((((((t1243 + t1244) + t1245) + t1246) + t1247) + t1248), 1.0e-06_c_double)
    t1268 = min(t1265, 0.88_c_double)
    t998 = (t31 + (t995 * 0.025_c_double))
    t1285 = (1.0_c_double - t1268)
    t961 = min((max((t151 - ((((t763 * t811) + (t772 * t817)) + (t781 * t823)) * 0.025_c_double)), 0.0_c_double) * (1.0_c_double - (t938 * 0.0025000000000000005_c_double))), 1.0_c_double)
    t1297 = min(t961, 0.76_c_double)
    t1257 = max(((t30 + ((sum(((((((t1181 + t1188) + t1195) + t1202) + t1209) + t1216) * t938)) * 0.003_c_double) / (sum(t938) + 1.0e-06_c_double))) - (t975 * 0.0075_c_double)), 0.0_c_double)
    t1299 = (1.0_c_double - t1297)
    t1018 = min(max(((-t998) / 0.42_c_double), 0.0_c_double), 1.0_c_double)
    t1009 = min(max((((t33 + t998) - 0.5_c_double) / 5.0_c_double), 0.0_c_double), 1.0_c_double)
    t1422 = ((((((((((((((min(max(((((t1009 * 27.0_c_double) + 186.0_c_double) - (t1018 * 34.0_c_double)) + (t754 * 54.0_c_double)), 0.0_c_double), 255.0_c_double) * t1285) + (((((t1243 + t1244) + t1248) * 255.0_c_double) / t1265) * t1268)) * t1299) + (t1297 * 226.0_c_double)) * t1342) + (((t1328 * 58.0_c_double) + 102.0_c_double) * t1325)) * t1357) + (t1355 * 218.0_c_double)) * t1389) + (t1378 * 53.0_c_double)) * t1391) + (t1387 * 205.0_c_double)) * t1418) + (t1416 * 245.0_c_double))
    t1430 = ((((((((((((((min(max(((((t1009 * 16.0_c_double) + 232.0_c_double) + (t1018 * 15.0_c_double)) + (t754 * 20.0_c_double)), 0.0_c_double), 255.0_c_double) * t1285) + (((((t1246 + t1247) + t1248) * 255.0_c_double) / t1265) * t1268)) * t1299) + (t1297 * 62.0_c_double)) * t1342) + (((t1328 * 24.0_c_double) + 48.0_c_double) * t1325)) * t1357) + (t1355 * 255.0_c_double)) * t1389) + (t1378 * 103.0_c_double)) * t1391) + (t1387 * 252.0_c_double)) * t1418) + (t1416 * 255.0_c_double))
    t1426 = ((((((((((((((min(max((((((t1009 * 18.0_c_double) + 220.0_c_double) - (t1018 * 21.0_c_double)) + (t754 * 30.0_c_double)) + (min(abs(t995), 1.0_c_double) * 8.0_c_double)), 0.0_c_double), 255.0_c_double) * t1285) + (((((t1244 + t1245) + t1246) * 255.0_c_double) / t1265) * t1268)) * t1299) + (t1297 * 181.0_c_double)) * t1342) + (((t1328 * 42.0_c_double) + 72.0_c_double) * t1325)) * t1357) + (t1355 * 249.0_c_double)) * t1389) + (t1378 * 83.0_c_double)) * t1391) + (t1387 * 245.0_c_double)) * t1418) + (t1416 * 252.0_c_double))
    return
  end subroutine columnar_multifluid_rgb_step
  subroutine columnar_multifluid_rgb_step_control(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t1422, t1426, t1430, t998, t995, t306, t315, t294, t297, t466, t475, t454, t457, t616, t625, t604, t607, t890, t900, t910, t961, t969, t1257, t35, t1243, t1244, t1245, t1246, t1247, t1248) bind(C, name="columnar_multifluid_rgb_step_control")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_1
    integer(c_int), intent(in), value :: extent_16
    real(c_double), intent(in) :: t0(extent_16)
    real(c_double), intent(in) :: t1(extent_16)
    real(c_double), intent(in) :: t2(extent_16)
    real(c_double), intent(in) :: t3(extent_16)
    real(c_double), intent(in) :: t4(extent_16)
    real(c_double), intent(in) :: t5(extent_16)
    real(c_double), intent(in) :: t6(extent_16)
    real(c_double), intent(in) :: t7(extent_16)
    real(c_double), intent(in) :: t8(extent_16)
    real(c_double), intent(in) :: t9(extent_16)
    real(c_double), intent(in) :: t10(extent_16)
    real(c_double), intent(in) :: t11(extent_16)
    real(c_double), intent(in) :: t12(extent_16)
    real(c_double), intent(in) :: t13(extent_16)
    real(c_double), intent(in) :: t14(extent_16)
    real(c_double), intent(in) :: t15(extent_16)
    real(c_double), intent(in) :: t16(extent_16)
    real(c_double), intent(in) :: t17(extent_16)
    real(c_double), intent(in) :: t18(extent_16)
    real(c_double), intent(in) :: t19(extent_16)
    real(c_double), intent(in) :: t20(extent_16)
    real(c_double), intent(in) :: t21(extent_16)
    real(c_double), intent(in) :: t22(extent_16)
    real(c_double), intent(in) :: t23(extent_16)
    real(c_double), intent(in) :: t24(extent_16)
    real(c_double), intent(in) :: t25(extent_16)
    real(c_double), intent(in) :: t26(extent_16)
    real(c_double), intent(in) :: t27(extent_16)
    real(c_double), intent(in) :: t28(extent_16)
    real(c_double), intent(in) :: t29(extent_16)
    real(c_double), intent(in) :: t30(extent_16)
    real(c_double), intent(in) :: t31(extent_16)
    real(c_double), intent(in) :: t32(extent_16)
    real(c_double), intent(in) :: t33(extent_16)
    real(c_double), intent(out) :: t1422(extent_16)
    real(c_double), intent(out) :: t1426(extent_16)
    real(c_double), intent(out) :: t1430(extent_16)
    real(c_double), intent(out) :: t998(extent_16)
    real(c_double), intent(out) :: t995(extent_16)
    real(c_double), intent(out) :: t306(extent_16)
    real(c_double), intent(out) :: t315(extent_16)
    real(c_double), intent(out) :: t294(extent_16)
    real(c_double), intent(out) :: t297(extent_16)
    real(c_double), intent(out) :: t466(extent_16)
    real(c_double), intent(out) :: t475(extent_16)
    real(c_double), intent(out) :: t454(extent_16)
    real(c_double), intent(out) :: t457(extent_16)
    real(c_double), intent(out) :: t616(extent_16)
    real(c_double), intent(out) :: t625(extent_16)
    real(c_double), intent(out) :: t604(extent_16)
    real(c_double), intent(out) :: t607(extent_16)
    real(c_double), intent(out) :: t890(extent_16)
    real(c_double), intent(out) :: t900(extent_16)
    real(c_double), intent(out) :: t910(extent_16)
    real(c_double), intent(out) :: t961(extent_16)
    real(c_double), intent(out) :: t969(extent_16)
    real(c_double), intent(out) :: t1257(extent_16)
    real(c_double), intent(out) :: t35(extent_16)
    real(c_double), intent(out) :: t1243(extent_16)
    real(c_double), intent(out) :: t1244(extent_16)
    real(c_double), intent(out) :: t1245(extent_16)
    real(c_double), intent(out) :: t1246(extent_16)
    real(c_double), intent(out) :: t1247(extent_16)
    real(c_double), intent(out) :: t1248(extent_16)

    ! block entry
    call numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t35, t294, t297, t454, t457, t604, t607, t306, t315, t466, t475, t616, t625, t1243, t1244, t1245, t1246, t1247, t1248, t969, t890, t900, t910, t995, t998, t961, t1257, t1422, t1430, t1426)
    return
  end subroutine columnar_multifluid_rgb_step_control
  subroutine numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t35, t294, t297, t454, t457, t604, t607, t306, t315, t466, t475, t616, t625, t1243, t1244, t1245, t1246, t1247, t1248, t969, t890, t900, t910, t995, t998, t961, t1257, t1422, t1430, t1426) bind(C, name="numerical_region_0")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_1
    integer(c_int), intent(in), value :: extent_16
    real(c_double), intent(in) :: t0(extent_16)
    real(c_double), intent(in) :: t1(extent_16)
    real(c_double), intent(in) :: t2(extent_16)
    real(c_double), intent(in) :: t3(extent_16)
    real(c_double), intent(in) :: t4(extent_16)
    real(c_double), intent(in) :: t5(extent_16)
    real(c_double), intent(in) :: t6(extent_16)
    real(c_double), intent(in) :: t7(extent_16)
    real(c_double), intent(in) :: t8(extent_16)
    real(c_double), intent(in) :: t9(extent_16)
    real(c_double), intent(in) :: t10(extent_16)
    real(c_double), intent(in) :: t11(extent_16)
    real(c_double), intent(in) :: t12(extent_16)
    real(c_double), intent(in) :: t13(extent_16)
    real(c_double), intent(in) :: t14(extent_16)
    real(c_double), intent(in) :: t15(extent_16)
    real(c_double), intent(in) :: t16(extent_16)
    real(c_double), intent(in) :: t17(extent_16)
    real(c_double), intent(in) :: t18(extent_16)
    real(c_double), intent(in) :: t19(extent_16)
    real(c_double), intent(in) :: t20(extent_16)
    real(c_double), intent(in) :: t21(extent_16)
    real(c_double), intent(in) :: t22(extent_16)
    real(c_double), intent(in) :: t23(extent_16)
    real(c_double), intent(in) :: t24(extent_16)
    real(c_double), intent(in) :: t25(extent_16)
    real(c_double), intent(in) :: t26(extent_16)
    real(c_double), intent(in) :: t27(extent_16)
    real(c_double), intent(in) :: t28(extent_16)
    real(c_double), intent(in) :: t29(extent_16)
    real(c_double), intent(in) :: t30(extent_16)
    real(c_double), intent(in) :: t31(extent_16)
    real(c_double), intent(in) :: t32(extent_16)
    real(c_double), intent(in) :: t33(extent_16)
    real(c_double), intent(out) :: t35(extent_16)
    real(c_double), intent(out) :: t294(extent_16)
    real(c_double), intent(out) :: t297(extent_16)
    real(c_double), intent(out) :: t454(extent_16)
    real(c_double), intent(out) :: t457(extent_16)
    real(c_double), intent(out) :: t604(extent_16)
    real(c_double), intent(out) :: t607(extent_16)
    real(c_double), intent(out) :: t306(extent_16)
    real(c_double), intent(out) :: t315(extent_16)
    real(c_double), intent(out) :: t466(extent_16)
    real(c_double), intent(out) :: t475(extent_16)
    real(c_double), intent(out) :: t616(extent_16)
    real(c_double), intent(out) :: t625(extent_16)
    real(c_double), intent(out) :: t1243(extent_16)
    real(c_double), intent(out) :: t1244(extent_16)
    real(c_double), intent(out) :: t1245(extent_16)
    real(c_double), intent(out) :: t1246(extent_16)
    real(c_double), intent(out) :: t1247(extent_16)
    real(c_double), intent(out) :: t1248(extent_16)
    real(c_double), intent(out) :: t969(extent_16)
    real(c_double), intent(out) :: t890(extent_16)
    real(c_double), intent(out) :: t900(extent_16)
    real(c_double), intent(out) :: t910(extent_16)
    real(c_double), intent(out) :: t995(extent_16)
    real(c_double), intent(out) :: t998(extent_16)
    real(c_double), intent(out) :: t961(extent_16)
    real(c_double), intent(out) :: t1257(extent_16)
    real(c_double), intent(out) :: t1422(extent_16)
    real(c_double), intent(out) :: t1430(extent_16)
    real(c_double), intent(out) :: t1426(extent_16)
    real(c_double) :: t34(extent_1)
    real(c_double) :: t36(extent_16)
    real(c_double) :: t37(extent_1)
    real(c_double) :: t50(extent_1)
    real(c_double) :: t52(extent_1)
    real(c_double) :: t55(extent_1)
    real(c_double) :: t57(extent_1)
    real(c_double) :: t73(extent_16)
    real(c_double) :: t74(extent_16)
    real(c_double) :: t72(extent_16)
    real(c_double) :: t76(extent_1)
    real(c_double) :: t105(extent_1)
    real(c_double) :: t107(extent_1)
    real(c_double) :: t115(extent_1)
    real(c_double) :: t117(extent_1)
    real(c_double) :: t152(extent_16)
    real(c_double) :: t153(extent_16)
    real(c_double) :: t207(extent_16)
    real(c_double) :: t208(extent_16)
    real(c_double) :: t205(extent_16)
    real(c_double) :: t206(extent_16)
    real(c_double) :: t99(extent_1)
    real(c_double) :: t219(extent_16)
    real(c_double) :: t100(extent_1)
    real(c_double) :: t220(extent_16)
    real(c_double) :: t257(extent_1)
    real(c_double) :: t285(extent_1)
    real(c_double) :: t316(extent_16)
    real(c_double) :: t317(extent_16)
    real(c_double) :: t372(extent_16)
    real(c_double) :: t373(extent_16)
    real(c_double) :: t379(extent_16)
    real(c_double) :: t380(extent_16)
    real(c_double) :: t476(extent_16)
    real(c_double) :: t477(extent_16)
    real(c_double) :: t529(extent_16)
    real(c_double) :: t530(extent_16)
    real(c_double) :: t626(extent_1)
    real(c_double) :: t628(extent_1)
    real(c_double) :: t183(extent_1)
    real(c_double) :: t634(extent_1)
    real(c_double) :: t918(extent_1)
    real(c_double) :: t919(extent_16)
    real(c_double) :: t920(extent_1)
    real(c_double) :: t921(extent_16)
    real(c_double) :: t922(extent_1)
    real(c_double) :: t923(extent_16)
    real(c_double) :: t990(extent_1)
    real(c_double) :: t1312(extent_16)
    real(c_double) :: t1313(extent_16)
    real(c_double) :: t79(extent_1)
    real(c_double) :: t80(extent_16)
    real(c_double) :: t86(extent_16)
    real(c_double) :: t92(extent_16)
    real(c_double) :: t101(extent_1)
    real(c_double) :: t139(extent_1)
    real(c_double) :: t176(extent_1)
    real(c_double) :: t236(extent_1)
    real(c_double) :: t268(extent_1)
    real(c_double) :: t396(extent_1)
    real(c_double) :: t428(extent_1)
    real(c_double) :: t546(extent_1)
    real(c_double) :: t578(extent_1)
    real(c_double) :: t975(extent_16)
    real(c_double) :: t1131(extent_1)
    real(c_double) :: t1139(extent_1)
    real(c_double) :: t1149(extent_1)
    real(c_double) :: t1162(extent_1)
    real(c_double) :: t1168(extent_1)
    real(c_double) :: t40(extent_16)
    real(c_double) :: t94(extent_16)
    real(c_double) :: t96(extent_16)
    real(c_double) :: t98(extent_16)
    real(c_double) :: t104(extent_16)
    real(c_double) :: t156(extent_16)
    real(c_double) :: t230(extent_1)
    real(c_double) :: t238(extent_1)
    real(c_double) :: t270(extent_1)
    real(c_double) :: t320(extent_16)
    real(c_double) :: t398(extent_1)
    real(c_double) :: t430(extent_1)
    real(c_double) :: t480(extent_16)
    real(c_double) :: t548(extent_1)
    real(c_double) :: t926(extent_16)
    real(c_double) :: t929(extent_16)
    real(c_double) :: t983(extent_1)
    real(c_double) :: t1141(extent_1)
    real(c_double) :: t1251(extent_1)
    real(c_double) :: t62(extent_1)
    real(c_double) :: t110(extent_1)
    real(c_double) :: t120(extent_1)
    real(c_double) :: t170(extent_1)
    real(c_double) :: t173(extent_1)
    real(c_double) :: t179(extent_1)
    real(c_double) :: t213(extent_16)
    real(c_double) :: t218(extent_16)
    real(c_double) :: t378(extent_16)
    real(c_double) :: t979(extent_1)
    real(c_double) :: t64(extent_16)
    real(c_double) :: t166(extent_1)
    real(c_double) :: t227(extent_16)
    real(c_double) :: t235(extent_1)
    real(c_double) :: t330(extent_1)
    real(c_double) :: t387(extent_16)
    real(c_double) :: t490(extent_1)
    real(c_double) :: t537(extent_16)
    real(c_double) :: t935(extent_1)
    real(c_double) :: t944(extent_1)
    real(c_double) :: t725(extent_1)
    real(c_double) :: t1136(extent_16)
    real(c_double) :: t1322(extent_1)
    real(c_double) :: t1375(extent_1)
    real(c_double) :: t1384(extent_1)
    real(c_double) :: t44(extent_1)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t68(extent_16)
    real(c_double) :: t938(extent_16)
    real(c_double) :: t947(extent_16)
    real(c_double) :: t1146(extent_16)
    real(c_double) :: t172(extent_16)
    real(c_double) :: t336(extent_16)
    real(c_double) :: t496(extent_16)
    real(c_double) :: t1156(extent_16)
    real(c_double) :: t1325(extent_16)
    real(c_double) :: t1378(extent_16)
    real(c_double) :: t47(extent_16)
    real(c_double) :: t1387(extent_16)
    real(c_double) :: t644(extent_1)
    real(c_double) :: t955(extent_1)
    real(c_double) :: t1219(extent_1)
    real(c_double) :: t1342(extent_16)
    real(c_double) :: t1389(extent_16)
    real(c_double) :: t1391(extent_16)
    real(c_double) :: t1393(extent_1)
    real(c_double) :: t1397(extent_1)
    real(c_double) :: t1401(extent_1)
    real(c_double) :: t1405(extent_1)
    real(c_double) :: t1409(extent_1)
    real(c_double) :: t1413(extent_1)
    real(c_double) :: t49(extent_16)
    real(c_double) :: t48(extent_16)
    real(c_double) :: t646(extent_16)
    real(c_double) :: t189(extent_1)
    real(c_double) :: t1354(extent_1)
    real(c_double) :: t1355(extent_16)
    real(c_double) :: t1242(extent_16)
    real(c_double) :: t1357(extent_16)
    real(c_double) :: t1359(extent_1)
    real(c_double) :: t1363(extent_1)
    real(c_double) :: t1038(extent_1)
    real(c_double) :: t71(extent_16)
    real(c_double) :: t131(extent_1)
    real(c_double) :: t132(extent_16)
    real(c_double) :: t143(extent_1)
    real(c_double) :: t151(extent_16)
    real(c_double) :: t359(extent_16)
    real(c_double) :: t516(extent_16)
    real(c_double) :: t192(extent_16)
    real(c_double) :: t228(extent_1)
    real(c_double) :: t302(extent_1)
    real(c_double) :: t305(extent_1)
    real(c_double) :: t314(extent_1)
    real(c_double) :: t653(extent_16)
    real(c_double) :: t660(extent_16)
    real(c_double) :: t667(extent_16)
    real(c_double) :: t728(extent_1)
    real(c_double) :: t737(extent_1)
    real(c_double) :: t746(extent_1)
    real(c_double) :: t760(extent_1)
    real(c_double) :: t769(extent_1)
    real(c_double) :: t778(extent_1)
    real(c_double) :: t850(extent_1)
    real(c_double) :: t859(extent_1)
    real(c_double) :: t868(extent_1)
    real(c_double) :: t1080(extent_1)
    real(c_double) :: t1089(extent_1)
    real(c_double) :: t1098(extent_1)
    real(c_double) :: t1107(extent_1)
    real(c_double) :: t1116(extent_1)
    real(c_double) :: t1125(extent_1)
    real(c_double) :: t772(extent_16)
    real(c_double) :: t763(extent_16)
    real(c_double) :: t781(extent_16)
    real(c_double) :: t1171(extent_1)
    real(c_double) :: t1174(extent_1)
    real(c_double) :: t722(extent_16)
    real(c_double) :: t873(extent_1)
    real(c_double) :: t874(extent_16)
    real(c_double) :: t877(extent_16)
    real(c_double) :: t880(extent_16)
    real(c_double) :: t754(extent_16)
    real(c_double) :: t1416(extent_16)
    real(c_double) :: t807(extent_1)
    real(c_double) :: t808(extent_1)
    real(c_double) :: t814(extent_1)
    real(c_double) :: t820(extent_1)
    real(c_double) :: t977(extent_1)
    real(c_double) :: t1031(extent_1)
    real(c_double) :: t1047(extent_1)
    real(c_double) :: t987(extent_1)
    real(c_double) :: t1181(extent_16)
    real(c_double) :: t1188(extent_16)
    real(c_double) :: t1195(extent_16)
    real(c_double) :: t1202(extent_16)
    real(c_double) :: t1209(extent_16)
    real(c_double) :: t1216(extent_16)
    real(c_double) :: t1418(extent_16)
    real(c_double) :: t810(extent_1)
    real(c_double) :: t811(extent_16)
    real(c_double) :: t817(extent_16)
    real(c_double) :: t823(extent_16)
    real(c_double) :: t1328(extent_16)
    real(c_double) :: t1330(extent_1)
    real(c_double) :: t1334(extent_1)
    real(c_double) :: t1338(extent_1)
    real(c_double) :: t1329(extent_1)
    real(c_double) :: t1333(extent_1)
    real(c_double) :: t1337(extent_1)
    real(c_double) :: t1265(extent_16)
    real(c_double) :: t1234(extent_1)
    real(c_double) :: t1267(extent_1)
    real(c_double) :: t1268(extent_16)
    real(c_double) :: t1285(extent_16)
    real(c_double) :: t1050(extent_1)
    real(c_double) :: t1000(extent_1)
    real(c_double) :: t1296(extent_1)
    real(c_double) :: t1297(extent_16)
    real(c_double) :: t1299(extent_16)
    real(c_double) :: t1301(extent_1)
    real(c_double) :: t1305(extent_1)
    real(c_double) :: t1309(extent_1)
    real(c_double) :: t1018(extent_16)
    real(c_double) :: t1009(extent_16)
    real(c_double) :: t1028(extent_1)
    real(c_double) :: t1044(extent_1)
    real(c_double) :: t1063(extent_1)
    real(c_double) :: t1025(extent_1)
    real(c_double) :: t1041(extent_1)
    real(c_double) :: t1060(extent_1)
    real(c_double) :: t1024(extent_1)
    real(c_double) :: t1040(extent_1)
    real(c_double) :: t1059(extent_1)

    ! block entry
    t34 = 0.025_c_double
    t35 = (t0 + t34(1))
    t36 = (t1 - t2)
    t37 = 2.0_c_double
    t50 = 0.61_c_double
    t52 = 0.83_c_double
    t55 = 0.37_c_double
    t57 = 0.29_c_double
    t73 = (t8 + t9)
    t74 = (t10 + t11)
    t72 = (t6 + t7)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t105 = 12.9898_c_double
    t107 = 78.233_c_double
    t115 = 39.3467_c_double
    t117 = 11.135_c_double
    t152 = (t4 - t16)
    t153 = (t5 - t17)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t207 = (t16 - t20)
    t208 = (t17 - t21)
    t205 = (t16 - t18)
    t206 = (t17 - t19)
    t99 = 5.0_c_double
    t219 = (t99(1) - t16)
    t100 = 3.45_c_double
    t220 = (t100(1) - t17)
    t99 = 5.0_c_double
    t257 = 0.54_c_double
    t285 = 3.5_c_double
    t257 = 0.54_c_double
    t316 = (t4 - t18)
    t317 = (t5 - t19)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t372 = (t18 - t20)
    t373 = (t19 - t21)
    t99 = 5.0_c_double
    t379 = (t99(1) - t18)
    t100 = 3.45_c_double
    t380 = (t100(1) - t19)
    t99 = 5.0_c_double
    t257 = 0.54_c_double
    t285 = 3.5_c_double
    t257 = 0.54_c_double
    t476 = (t4 - t20)
    t477 = (t5 - t21)
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t99 = 5.0_c_double
    t529 = (t99(1) - t20)
    t100 = 3.45_c_double
    t530 = (t100(1) - t21)
    t99 = 5.0_c_double
    t257 = 0.54_c_double
    t285 = 3.5_c_double
    t257 = 0.54_c_double
    t626 = 0.42_c_double
    t628 = 0.34_c_double
    t183 = 0.18_c_double
    t634 = 0.35_c_double
    t918 = 1.15_c_double
    t919 = (t4 - t918(1))
    t920 = 5.75_c_double
    t921 = (t5 - t920(1))
    t922 = 8.85_c_double
    t923 = (t4 - t922(1))
    t76 = 0.0_c_double
    t990 = 4.6_c_double
    t99 = 5.0_c_double
    t1312 = (t4 - t99(1))
    t100 = 3.45_c_double
    t1313 = (t5 - t100(1))
    t79 = 1.0_c_double
    t80 = min(max(t12, t76(1)), t79(1))
    t79 = 1.0_c_double
    t86 = min(max(t13, t76(1)), t79(1))
    t79 = 1.0_c_double
    t92 = min(max(t14, t76(1)), t79(1))
    t101 = 0.08_c_double
    t139 = 0.9999250028124297_c_double
    t176 = 0.3_c_double
    t236 = 1.71_c_double
    t176 = 0.3_c_double
    t176 = 0.3_c_double
    t101 = 0.08_c_double
    t268 = 1.37_c_double
    t176 = 0.3_c_double
    t176 = 0.3_c_double
    t101 = 0.08_c_double
    t176 = 0.3_c_double
    t396 = 1.63_c_double
    t176 = 0.3_c_double
    t176 = 0.3_c_double
    t101 = 0.08_c_double
    t428 = 1.43_c_double
    t176 = 0.3_c_double
    t176 = 0.3_c_double
    t101 = 0.08_c_double
    t176 = 0.3_c_double
    t546 = 1.79_c_double
    t176 = 0.3_c_double
    t176 = 0.3_c_double
    t101 = 0.08_c_double
    t578 = 1.31_c_double
    t176 = 0.3_c_double
    t176 = 0.3_c_double
    t101 = 0.08_c_double
    t626 = 0.42_c_double
    t975 = min(max(t30, t76(1)), t626(1))
    t1131 = 2.11_c_double
    t1139 = 1.91_c_double
    t1149 = 2.27_c_double
    t1162 = 0.9976278180810777_c_double
    t1168 = 0.9982016190284373_c_double
    t1162 = 0.9976278180810777_c_double
    t1168 = 0.9982016190284373_c_double
    t1162 = 0.9976278180810777_c_double
    t1168 = 0.9982016190284373_c_double
    t40 = (((t3 * t37(1)) - t1) - t2)
    t79 = 1.0_c_double
    t94 = (t79(1) - t80)
    t79 = 1.0_c_double
    t96 = (t79(1) - t86)
    t79 = 1.0_c_double
    t98 = (t79(1) - t92)
    t104 = real(floor((t35 * t101(1))), c_double)
    t156 = ((t152 * t152) + (t153 * t153))
    t230 = 1.85_c_double
    t238 = 0.2_c_double
    t230 = 1.85_c_double
    t270 = 1.1_c_double
    t320 = ((t316 * t316) + (t317 * t317))
    t230 = 1.85_c_double
    t398 = 2.3_c_double
    t230 = 1.85_c_double
    t430 = 2.8_c_double
    t480 = ((t476 * t476) + (t477 * t477))
    t230 = 1.85_c_double
    t548 = 4.2_c_double
    t230 = 1.85_c_double
    t99 = 5.0_c_double
    t926 = ((t919 * t919) + (t921 * t921))
    t929 = ((t923 * t923) + (t921 * t921))
    t983 = 0.16_c_double
    t1141 = 2.1_c_double
    t548 = 4.2_c_double
    t1251 = 0.0075_c_double
    t62 = 0.72_c_double
    t110 = 37.719_c_double
    t120 = 19.913_c_double
    t101 = 0.08_c_double
    t170 = 0.14_c_double
    t173 = 1.2_c_double
    t179 = 1.65_c_double
    t183 = 0.18_c_double
    t213 = (((t205 * t205) + (t206 * t206)) + t183(1))
    t183 = 0.18_c_double
    t218 = (((t207 * t207) + (t208 * t208)) + t183(1))
    t101 = 0.08_c_double
    t101 = 0.08_c_double
    t170 = 0.14_c_double
    t173 = 1.2_c_double
    t179 = 1.65_c_double
    t183 = 0.18_c_double
    t378 = (((t372 * t372) + (t373 * t373)) + t183(1))
    t101 = 0.08_c_double
    t101 = 0.08_c_double
    t170 = 0.14_c_double
    t173 = 1.2_c_double
    t179 = 1.65_c_double
    t101 = 0.08_c_double
    t979 = 0.22_c_double
    t64 = (((t4 * t50(1)) + (t5 * t52(1))) + (sin(((t4 * t55(1)) - (t5 * t57(1)))) * t62(1)))
    t166 = 4.805000000000001_c_double
    t227 = sqrt((((t219 * t219) + (t220 * t220)) + t101(1)))
    t235 = 0.58_c_double
    t235 = 0.58_c_double
    t330 = 4.805000000000001_c_double
    t387 = sqrt((((t379 * t379) + (t380 * t380)) + t101(1)))
    t235 = 0.58_c_double
    t235 = 0.58_c_double
    t490 = 4.805000000000001_c_double
    t537 = sqrt((((t529 * t529) + (t530 * t530)) + t101(1)))
    t235 = 0.58_c_double
    t235 = 0.58_c_double
    t76 = 0.0_c_double
    t935 = 0.4608_c_double
    t944 = 0.6728_c_double
    t725 = 0.78_c_double
    t1136 = ((sin((t35 * t1131(1))) * t979(1)) + t725(1))
    t979 = 0.22_c_double
    t979 = 0.22_c_double
    t1322 = 0.23120000000000004_c_double
    t1375 = 0.1058_c_double
    t1384 = 0.1058_c_double
    t44 = 1.0e-05_c_double
    t66 = cos(t64)
    t68 = sin(t64)
    t79 = 1.0_c_double
    t938 = exp(((-t926) / t935(1)))
    t947 = exp(((-t929) / t944(1)))
    t725 = 0.78_c_double
    t1146 = ((sin(((t35 * t1139(1)) + t1141(1))) * t979(1)) + t725(1))
    t725 = 0.78_c_double
    t172 = (exp(((-t156) / t166(1))) / (t156 + t170(1)))
    t336 = (exp(((-t320) / t330(1))) / (t320 + t170(1)))
    t496 = (exp(((-t480) / t490(1))) / (t480 + t170(1)))
    t1156 = ((sin(((t35 * t1149(1)) + t548(1))) * t979(1)) + t725(1))
    t1325 = exp(((-((t1312 * t1312) + (t1313 * t1313))) / t1322(1)))
    t1378 = exp(((-t926) / t1375(1)))
    t47 = sqrt((((t36 * t36) + (t40 * t40)) + t44(1)))
    t1387 = exp(((-t929) / t1384(1)))
    t644 = 0.12_c_double
    t955 = 0.0025000000000000005_c_double
    t1219 = 0.003_c_double
    t79 = 1.0_c_double
    t1342 = (t79(1) - t1325)
    t79 = 1.0_c_double
    t1389 = (t79(1) - t1378)
    t79 = 1.0_c_double
    t1391 = (t79(1) - t1387)
    t1393 = 53.0_c_double
    t1397 = 83.0_c_double
    t1401 = 103.0_c_double
    t1405 = 205.0_c_double
    t1409 = 245.0_c_double
    t1413 = 252.0_c_double
    t49 = (t40 / t47)
    t48 = (t36 / t47)
    t62 = 0.72_c_double
    t183 = 0.18_c_double
    t646 = ((min(max(((((t1 * t626(1)) + (t3 * t628(1))) + (t2 * t183(1))) + (t28 * t634(1))), t76(1)), t79(1)) * t644(1)) + t183(1))
    t79 = 1.0_c_double
    t101 = 0.08_c_double
    t189 = 1.0e-06_c_double
    t1354 = 0.6_c_double
    t1355 = min((t947 * t975), t1354(1))
    t76 = 0.0_c_double
    t189 = 1.0e-06_c_double
    t189 = 1.0e-06_c_double
    t189 = 1.0e-06_c_double
    t79 = 1.0_c_double
    t1242 = (t79(1) - min((t938 * t1219(1)), t101(1)))
    t79 = 1.0_c_double
    t1357 = (t79(1) - t1355)
    t1359 = 218.0_c_double
    t1363 = 249.0_c_double
    t1038 = 255.0_c_double
    t71 = ((t66 * t48) + (t68 * t49))
    t131 = 0.28_c_double
    t132 = (max(((sin((((t4 * t105(1)) + (t5 * t107(1))) + (t104 * t110(1)))) * cos((((t4 * t115(1)) - (t5 * t117(1))) + (t104 * t120(1))))) - t62(1)), t76(1)) / t131(1))
    t183 = 0.18_c_double
    t183 = 0.18_c_double
    t183 = 0.18_c_double
    t143 = 0.0008_c_double
    t76 = 0.0_c_double
    t79 = 1.0_c_double
    t151 = min(max(((t15 * t139(1)) + ((t132 * t132) * t143(1))), t76(1)), t79(1))
    t359 = ((t336 * (((((t96 * t173(1)) * t72) - (t73 * t176(1))) + ((t96 * t179(1)) * t151)) + (((t68 * t48) - (t66 * t49)) * t183(1)))) / (sqrt((t320 + t101(1))) * (sum(t336) + t189(1))))
    t516 = ((t496 * (((((t98 * t173(1)) * t73) - (t74 * t176(1))) + ((t98 * t179(1)) * t151)) - (t71 * t183(1)))) / (sqrt((t480 + t101(1))) * (sum(t496) + t189(1))))
    t192 = ((t172 * (((((t94 * t173(1)) * t74) - (t72 * t176(1))) + ((t94 * t179(1)) * t151)) + (t71 * t183(1)))) / (sqrt((t156 + t101(1))) * (sum(t172) + t189(1))))
    t228 = 2.35_c_double
    t228 = 2.35_c_double
    t228 = 2.35_c_double
    t228 = 2.35_c_double
    t228 = 2.35_c_double
    t228 = 2.35_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t294 = (t22 + ((((((((((t4 * t76(1)) + sum((t152 * t192))) * t228(1)) + (((t80 * t230(1)) * t219) / t227)) + (cos(((t35 * t236(1)) + t238(1))) * t235(1))) + ((t205 * t176(1)) / t213)) + ((t207 * t176(1)) / t218)) + ((t99(1) - t16) * t101(1))) - (t22 * t257(1))) * t34(1)))
    t297 = (t23 + ((((((((((t5 * t76(1)) + sum((t153 * t192))) * t228(1)) + (((t80 * t230(1)) * t220) / t227)) + (sin(((t35 * t268(1)) + t270(1))) * t235(1))) + ((t206 * t176(1)) / t213)) + ((t208 * t176(1)) / t218)) + ((t285(1) - t17) * t101(1))) - (t23 * t257(1))) * t34(1)))
    t454 = (t24 + ((((((((((t4 * t76(1)) + sum((t316 * t359))) * t228(1)) + (((t86 * t230(1)) * t379) / t387)) + (cos(((t35 * t396(1)) + t398(1))) * t235(1))) - ((t205 * t176(1)) / t213)) + ((t372 * t176(1)) / t378)) + ((t99(1) - t18) * t101(1))) - (t24 * t257(1))) * t34(1)))
    t604 = (t26 + ((((((((((t4 * t76(1)) + sum((t476 * t516))) * t228(1)) + (((t92 * t230(1)) * t529) / t537)) + (cos(((t35 * t546(1)) + t548(1))) * t235(1))) - ((t207 * t176(1)) / t218)) - ((t372 * t176(1)) / t378)) + ((t99(1) - t20) * t101(1))) - (t26 * t257(1))) * t34(1)))
    t457 = (t25 + ((((((((((t5 * t76(1)) + sum((t317 * t359))) * t228(1)) + (((t86 * t230(1)) * t380) / t387)) + (sin(((t35 * t428(1)) + t430(1))) * t235(1))) - ((t206 * t176(1)) / t213)) + ((t373 * t176(1)) / t378)) + ((t285(1) - t19) * t101(1))) - (t25 * t257(1))) * t34(1)))
    t607 = (t27 + ((((((((((t5 * t76(1)) + sum((t477 * t516))) * t228(1)) + (((t92 * t230(1)) * t530) / t537)) + (sin(((t35 * t578(1)) + t99(1))) * t235(1))) - ((t208 * t176(1)) / t218)) - ((t373 * t176(1)) / t378)) + ((t285(1) - t21) * t101(1))) - (t27 * t257(1))) * t34(1)))
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t302 = 0.65_c_double
    t302 = 0.65_c_double
    t302 = 0.65_c_double
    t302 = 0.65_c_double
    t302 = 0.65_c_double
    t302 = 0.65_c_double
    t305 = 9.35_c_double
    t306 = min(max((t16 + (t294 * t34(1))), t302(1)), t305(1))
    t314 = 6.35_c_double
    t315 = min(max((t17 + (t297 * t34(1))), t302(1)), t314(1))
    t305 = 9.35_c_double
    t466 = min(max((t18 + (t454 * t34(1))), t302(1)), t305(1))
    t314 = 6.35_c_double
    t475 = min(max((t19 + (t457 * t34(1))), t302(1)), t314(1))
    t305 = 9.35_c_double
    t616 = min(max((t20 + (t604 * t34(1))), t302(1)), t305(1))
    t314 = 6.35_c_double
    t625 = min(max((t21 + (t607 * t34(1))), t302(1)), t314(1))
    t99 = 5.0_c_double
    t99 = 5.0_c_double
    t100 = 3.45_c_double
    t100 = 3.45_c_double
    t99 = 5.0_c_double
    t99 = 5.0_c_double
    t100 = 3.45_c_double
    t100 = 3.45_c_double
    t99 = 5.0_c_double
    t99 = 5.0_c_double
    t100 = 3.45_c_double
    t100 = 3.45_c_double
    t653 = (((t4 - t306) * (t4 - t306)) + ((t5 - t315) * (t5 - t315)))
    t660 = (((t4 - t466) * (t4 - t466)) + ((t5 - t475) * (t5 - t475)))
    t667 = (((t4 - t616) * (t4 - t616)) + ((t5 - t625) * (t5 - t625)))
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t728 = 1.2168_c_double
    t737 = 1.2168_c_double
    t746 = 1.2168_c_double
    t760 = 0.18_c_double
    t769 = 0.18_c_double
    t778 = 0.18_c_double
    t850 = 0.32000000000000006_c_double
    t859 = 0.32000000000000006_c_double
    t868 = 0.32000000000000006_c_double
    t1080 = 0.1152_c_double
    t1089 = 0.2888_c_double
    t1098 = 0.1152_c_double
    t1107 = 0.2888_c_double
    t1116 = 0.1152_c_double
    t1125 = 0.2888_c_double
    t772 = exp(((-t660) / t769(1)))
    t763 = exp(((-t653) / t760(1)))
    t781 = exp(((-t667) / t778(1)))
    t1171 = 0.08000000000000002_c_double
    t1174 = 0.05500000000000001_c_double
    t1171 = 0.08000000000000002_c_double
    t1174 = 0.05500000000000001_c_double
    t1171 = 0.08000000000000002_c_double
    t1174 = 0.05500000000000001_c_double
    t722 = max(max((min(max((t646 - abs((t4 - t306))), t76(1)), max((t646 - abs((t5 - t315))), t76(1))) / t646), (min(max((t646 - abs((t4 - t466))), t76(1)), max((t646 - abs((t5 - t475))), t76(1))) / t646)), (min(max((t646 - abs((t4 - t616))), t76(1)), max((t646 - abs((t5 - t625))), t76(1))) / t646))
    t189 = 1.0e-06_c_double
    t189 = 1.0e-06_c_double
    t189 = 1.0e-06_c_double
    t873 = 2.4_c_double
    t874 = ((t80 * exp(((-(((t306 - t99(1)) * (t306 - t99(1))) + ((t315 - t100(1)) * (t315 - t100(1))))) / t850(1)))) * t873(1))
    t873 = 2.4_c_double
    t877 = ((t86 * exp(((-(((t466 - t99(1)) * (t466 - t99(1))) + ((t475 - t100(1)) * (t475 - t100(1))))) / t859(1)))) * t873(1))
    t873 = 2.4_c_double
    t880 = ((t92 * exp(((-(((t616 - t99(1)) * (t616 - t99(1))) + ((t625 - t100(1)) * (t625 - t100(1))))) / t868(1)))) * t873(1))
    t79 = 1.0_c_double
    t754 = min(((exp(((-t653) / t728(1))) + exp(((-t660) / t737(1)))) + exp(((-t667) / t746(1)))), t79(1))
    t979 = 0.22_c_double
    t1416 = (t722 * t722)
    t807 = 0.7_c_double
    t808 = min((sum((t151 * t763)) / (sum(t763) + t189(1))), t807)
    t807 = 0.7_c_double
    t814 = min((sum((t151 * t772)) / (sum(t772) + t189(1))), t807)
    t807 = 0.7_c_double
    t820 = min((sum((t151 * t781)) / (sum(t781) + t189(1))), t807)
    t977 = -0.42_c_double
    t1031 = 54.0_c_double
    t1047 = 30.0_c_double
    t987 = 20.0_c_double
    t79 = 1.0_c_double
    t1181 = min(((t6 * t1162(1)) + ((exp(((-t653) / t1080(1))) * t1171(1)) * t1136)), t79(1))
    t79 = 1.0_c_double
    t1188 = min(((t7 * t1168(1)) + ((exp(((-t653) / t1089(1))) * t1174(1)) * t1136)), t79(1))
    t79 = 1.0_c_double
    t1195 = min(((t8 * t1162(1)) + ((exp(((-t660) / t1098(1))) * t1171(1)) * t1146)), t79(1))
    t79 = 1.0_c_double
    t1202 = min(((t9 * t1168(1)) + ((exp(((-t660) / t1107(1))) * t1174(1)) * t1146)), t79(1))
    t79 = 1.0_c_double
    t1209 = min(((t10 * t1162(1)) + ((exp(((-t667) / t1116(1))) * t1171(1)) * t1156)), t79(1))
    t79 = 1.0_c_double
    t1216 = min(((t11 * t1168(1)) + ((exp(((-t667) / t1125(1))) * t1174(1)) * t1156)), t79(1))
    t79 = 1.0_c_double
    t1418 = (t79(1) - t1416)
    t1409 = 245.0_c_double
    t1413 = 252.0_c_double
    t1038 = 255.0_c_double
    t34 = 0.025_c_double
    t1243 = (t1181 * t1242)
    t1244 = (t1188 * t1242)
    t1246 = (t1202 * t1242)
    t1248 = (t1216 * t1242)
    t1245 = (t1195 * t1242)
    t1247 = (t1209 * t1242)
    t810 = 1.35_c_double
    t811 = ((t94 * t808(1)) * t810(1))
    t810 = 1.35_c_double
    t817 = ((t96 * t814(1)) * t810(1))
    t810 = 1.35_c_double
    t823 = ((t98 * t820(1)) * t810(1))
    t76 = 0.0_c_double
    t969 = max((t29 + (((t874 + t877) + t880) * t34(1))), t76(1))
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t987 = 20.0_c_double
    t1038 = 255.0_c_double
    t1038 = 255.0_c_double
    t1038 = 255.0_c_double
    t79 = 1.0_c_double
    t1328 = min(t969, t79(1))
    t1330 = 58.0_c_double
    t1334 = 42.0_c_double
    t1338 = 24.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t34 = 0.025_c_double
    t34 = 0.025_c_double
    t1329 = 102.0_c_double
    t1333 = 72.0_c_double
    t1337 = 48.0_c_double
    t79 = 1.0_c_double
    t890 = min(max((t80 + ((t811 - t874) * t34(1))), t76(1)), t79(1))
    t79 = 1.0_c_double
    t900 = min(max((t86 + ((t817 - t877) * t34(1))), t76(1)), t79(1))
    t79 = 1.0_c_double
    t995 = (t32 + (((((((t754 * t977(1)) - ((t722 * t979(1)) * t722)) + ((t975 * t983(1)) * t947)) - t31) * t987(1)) - (t32 * t990(1))) * t34(1)))
    t910 = min(max((t92 + ((t823 - t880) * t34(1))), t76(1)), t79(1))
    t189 = 1.0e-06_c_double
    t1265 = max((((((t1243 + t1244) + t1245) + t1246) + t1247) + t1248), t189(1))
    t76 = 0.0_c_double
    t34 = 0.025_c_double
    t1234 = 0.003_c_double
    t1267 = 0.88_c_double
    t1268 = min(t1265, t1267(1))
    t998 = (t31 + (t995 * t34(1)))
    t79 = 1.0_c_double
    t79 = 1.0_c_double
    t1285 = (t79(1) - t1268)
    t79 = 1.0_c_double
    t961 = min((max((t151 - ((((t763 * t811) + (t772 * t817)) + (t781 * t823)) * t34(1))), t76(1)) * (t79(1) - (t938 * t955(1)))), t79(1))
    t1050 = 8.0_c_double
    t1000 = 0.5_c_double
    t626 = 0.42_c_double
    t1296 = 0.76_c_double
    t1297 = min(t961, t1296(1))
    t99 = 5.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t1257 = max(((t30 + ((sum(((((((t1181 + t1188) + t1195) + t1202) + t1209) + t1216) * t938)) * t1234(1)) / (sum(t938) + t189(1)))) - (t975 * t1251(1))), t76(1))
    t79 = 1.0_c_double
    t1299 = (t79(1) - t1297)
    t1301 = 226.0_c_double
    t1305 = 181.0_c_double
    t1309 = 62.0_c_double
    t76 = 0.0_c_double
    t79 = 1.0_c_double
    t1018 = min(max(((-t998) / t626(1)), t76(1)), t79(1))
    t79 = 1.0_c_double
    t1009 = min(max((((t33 + t998) - t1000(1)) / t99(1)), t76(1)), t79(1))
    t1028 = 34.0_c_double
    t1044 = 21.0_c_double
    t1063 = 15.0_c_double
    t1025 = 27.0_c_double
    t1041 = 18.0_c_double
    t1060 = 16.0_c_double
    t1024 = 186.0_c_double
    t1040 = 220.0_c_double
    t1059 = 232.0_c_double
    t76 = 0.0_c_double
    t76 = 0.0_c_double
    t1038 = 255.0_c_double
    t76 = 0.0_c_double
    t1038 = 255.0_c_double
    t1038 = 255.0_c_double
    t1422 = ((((((((((((((min(max(((((t1009 * t1025(1)) + t1024(1)) - (t1018 * t1028(1))) + (t754 * t1031(1))), t76(1)), t1038(1)) * t1285) + (((((t1243 + t1244) + t1248) * t1038(1)) / t1265) * t1268)) * t1299) + (t1297 * t1301(1))) * t1342) + (((t1328 * t1330(1)) + t1329(1)) * t1325)) * t1357) + (t1355 * t1359(1))) * t1389) + (t1378 * t1393(1))) * t1391) + (t1387 * t1405(1))) * t1418) + (t1416 * t1409(1)))
    t1430 = ((((((((((((((min(max(((((t1009 * t1060(1)) + t1059(1)) + (t1018 * t1063(1))) + (t754 * t987(1))), t76(1)), t1038(1)) * t1285) + (((((t1246 + t1247) + t1248) * t1038(1)) / t1265) * t1268)) * t1299) + (t1297 * t1309(1))) * t1342) + (((t1328 * t1338(1)) + t1337(1)) * t1325)) * t1357) + (t1355 * t1038(1))) * t1389) + (t1378 * t1401(1))) * t1391) + (t1387 * t1413(1))) * t1418) + (t1416 * t1038(1)))
    t1426 = ((((((((((((((min(max((((((t1009 * t1041(1)) + t1040(1)) - (t1018 * t1044(1))) + (t754 * t1047(1))) + (min(abs(t995), t79(1)) * t1050(1))), t76(1)), t1038(1)) * t1285) + (((((t1244 + t1245) + t1246) * t1038(1)) / t1265) * t1268)) * t1299) + (t1297 * t1305(1))) * t1342) + (((t1328 * t1334(1)) + t1333(1)) * t1325)) * t1357) + (t1355 * t1363(1))) * t1389) + (t1378 * t1397(1))) * t1391) + (t1387 * t1409(1))) * t1418) + (t1416 * t1413(1)))
    return
  end subroutine numerical_region_0

end module kernel_fortran
