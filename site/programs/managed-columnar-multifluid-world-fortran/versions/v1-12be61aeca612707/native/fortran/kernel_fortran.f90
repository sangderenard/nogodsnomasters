module kernel_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains

  subroutine columnar_multifluid_rgb_step(extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t900, t906, t912, t610, t608, t198, t206, t188, t190, t332, t340, t322, t324, t456, t464, t446, t448, t29, t782, t796, t810, t824, t838, t852) bind(C, name="columnar_multifluid_rgb_step")
    use, intrinsic :: iso_c_binding
    implicit none
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
    real(c_double), intent(out) :: t900(extent_16)
    real(c_double), intent(out) :: t906(extent_16)
    real(c_double), intent(out) :: t912(extent_16)
    real(c_double), intent(out) :: t610(extent_16)
    real(c_double), intent(out) :: t608(extent_16)
    real(c_double), intent(out) :: t198(extent_16)
    real(c_double), intent(out) :: t206(extent_16)
    real(c_double), intent(out) :: t188(extent_16)
    real(c_double), intent(out) :: t190(extent_16)
    real(c_double), intent(out) :: t332(extent_16)
    real(c_double), intent(out) :: t340(extent_16)
    real(c_double), intent(out) :: t322(extent_16)
    real(c_double), intent(out) :: t324(extent_16)
    real(c_double), intent(out) :: t456(extent_16)
    real(c_double), intent(out) :: t464(extent_16)
    real(c_double), intent(out) :: t446(extent_16)
    real(c_double), intent(out) :: t448(extent_16)
    real(c_double), intent(out) :: t29(extent_16)
    real(c_double), intent(out) :: t782(extent_16)
    real(c_double), intent(out) :: t796(extent_16)
    real(c_double), intent(out) :: t810(extent_16)
    real(c_double), intent(out) :: t824(extent_16)
    real(c_double), intent(out) :: t838(extent_16)
    real(c_double), intent(out) :: t852(extent_16)
    real(c_double) :: t30(extent_16)
    real(c_double) :: t68(extent_16)
    real(c_double) :: t67(extent_16)
    real(c_double) :: t70(extent_16)
    real(c_double) :: t69(extent_16)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t119(extent_16)
    real(c_double) :: t122(extent_16)
    real(c_double) :: t120(extent_16)
    real(c_double) :: t121(extent_16)
    real(c_double) :: t208(extent_16)
    real(c_double) :: t207(extent_16)
    real(c_double) :: t260(extent_16)
    real(c_double) :: t261(extent_16)
    real(c_double) :: t341(extent_16)
    real(c_double) :: t342(extent_16)
    real(c_double) :: t34(extent_16)
    real(c_double) :: t73(extent_16)
    real(c_double) :: t211(extent_16)
    real(c_double) :: t345(extent_16)
    real(c_double) :: t127(extent_16)
    real(c_double) :: t132(extent_16)
    real(c_double) :: t266(extent_16)
    real(c_double) :: t58(extent_16)
    real(c_double) :: t77(extent_16)
    real(c_double) :: t215(extent_16)
    real(c_double) :: t349(extent_16)
    real(c_double) :: t748(extent_16)
    real(c_double) :: t60(extent_16)
    real(c_double) :: t62(extent_16)
    real(c_double) :: t758(extent_16)
    real(c_double) :: t227(extent_16)
    real(c_double) :: t768(extent_16)
    real(c_double) :: t89(extent_16)
    real(c_double) :: t41(extent_16)
    real(c_double) :: t361(extent_16)
    real(c_double) :: t42(extent_16)
    real(c_double) :: t43(extent_16)
    real(c_double) :: t485(extent_16)
    real(c_double) :: t102
    real(c_double) :: t243
    real(c_double) :: t374
    real(c_double) :: t65(extent_16)
    real(c_double) :: t239(extent_16)
    real(c_double) :: t98(extent_16)
    real(c_double) :: t370(extent_16)
    real(c_double) :: t492(extent_16)
    real(c_double) :: t499(extent_16)
    real(c_double) :: t506(extent_16)
    real(c_double) :: t561(extent_16)
    real(c_double) :: t593(extent_16)
    real(c_double) :: t894(extent_16)
    real(c_double) :: t860(extent_16)
    real(c_double) :: t863(extent_16)
    real(c_double) :: t630(extent_16)
    real(c_double) :: t621(extent_16)

    ! block entry
    t30 = (t2 - t3)
    t29 = (t0 + t1)
    t68 = (t11 + t12)
    t67 = (t9 + t10)
    t70 = (t6 - t14)
    t69 = (t5 - t13)
    t66 = (t7 + t8)
    t119 = (t13 - t15)
    t122 = (t14 - t18)
    t120 = (t14 - t16)
    t121 = (t13 - t17)
    t208 = (t6 - t16)
    t207 = (t5 - t15)
    t260 = (t15 - t17)
    t261 = (t16 - t18)
    t341 = (t5 - t17)
    t342 = (t6 - t18)
    t34 = (((t4 * 2.0_c_double) - t2) - t3)
    t73 = ((t69 * t69) + (t70 * t70))
    t211 = ((t207 * t207) + (t208 * t208))
    t345 = ((t341 * t341) + (t342 * t342))
    t127 = (((t119 * t119) + (t120 * t120)) + 0.18_c_double)
    t132 = (((t121 * t121) + (t122 * t122)) + 0.18_c_double)
    t266 = (((t260 * t260) + (t261 * t261)) + 0.18_c_double)
    t58 = (((t5 * 0.61_c_double) + (t6 * 0.83_c_double)) + (sin(((t5 * 0.37_c_double) - (t6 * 0.29_c_double))) * 0.72_c_double))
    t77 = sqrt((t73 + 0.08_c_double))
    t215 = sqrt((t211 + 0.08_c_double))
    t349 = sqrt((t345 + 0.08_c_double))
    t748 = ((sin((t29 * 2.11_c_double)) * 0.22_c_double) + 0.78_c_double)
    t60 = cos(t58)
    t62 = sin(t58)
    t758 = ((sin(((t29 * 1.91_c_double) + 2.1_c_double)) * 0.22_c_double) + 0.78_c_double)
    t227 = (exp(((-t211) / 4.805000000000001_c_double)) / (t211 + 0.14_c_double))
    t768 = ((sin(((t29 * 2.27_c_double) + 4.2_c_double)) * 0.22_c_double) + 0.78_c_double)
    t89 = (exp(((-t73) / 4.805000000000001_c_double)) / (t73 + 0.14_c_double))
    t41 = sqrt((((t30 * t30) + (t34 * t34)) + 1.0e-05_c_double))
    t361 = (exp(((-t345) / 4.805000000000001_c_double)) / (t345 + 0.14_c_double))
    t42 = (t30 / t41)
    t43 = (t34 / t41)
    t485 = ((min(max(((((t2 * 0.42_c_double) + (t4 * 0.34_c_double)) + (t3 * 0.18_c_double)) + (t25 * 0.35_c_double)), 0.0_c_double), 1.0_c_double) * 0.12_c_double) + 0.18_c_double)
    t102 = (sum(t89) + 1.0e-06_c_double)
    t243 = (sum(t227) + 1.0e-06_c_double)
    t374 = (sum(t361) + 1.0e-06_c_double)
    t65 = ((t60 * t42) + (t62 * t43))
    t239 = (t227 * (((t66 * 1.35_c_double) - (t67 * 0.3_c_double)) + (((t62 * t42) - (t60 * t43)) * 0.18_c_double)))
    t98 = (t89 * (((t68 * 1.35_c_double) - (t66 * 0.3_c_double)) + (t65 * 0.18_c_double)))
    t370 = (t361 * (((t67 * 1.35_c_double) - (t68 * 0.3_c_double)) - (t65 * 0.18_c_double)))
    t188 = (t19 + (((((((((t5 * 0.0_c_double) + (sum(((t69 / t77) * t98)) / t102)) * 2.35_c_double) + (cos(((t29 * 1.71_c_double) + 0.2_c_double)) * 0.58_c_double)) + ((t119 * 0.3_c_double) / t127)) + ((t121 * 0.3_c_double) / t132)) + ((5.0_c_double - t13) * 0.08_c_double)) - (t19 * 0.54_c_double)) * t1))
    t190 = (t20 + (((((((((t6 * 0.0_c_double) + (sum(((t70 / t77) * t98)) / t102)) * 2.35_c_double) + (sin(((t29 * 1.37_c_double) + 1.1_c_double)) * 0.58_c_double)) + ((t120 * 0.3_c_double) / t127)) + ((t122 * 0.3_c_double) / t132)) + ((3.5_c_double - t14) * 0.08_c_double)) - (t20 * 0.54_c_double)) * t1))
    t322 = (t21 + (((((((((t5 * 0.0_c_double) + (sum(((t207 / t215) * t239)) / t243)) * 2.35_c_double) + (cos(((t29 * 1.63_c_double) + 2.3_c_double)) * 0.58_c_double)) - ((t119 * 0.3_c_double) / t127)) + ((t260 * 0.3_c_double) / t266)) + ((5.0_c_double - t15) * 0.08_c_double)) - (t21 * 0.54_c_double)) * t1))
    t324 = (t22 + (((((((((t6 * 0.0_c_double) + (sum(((t208 / t215) * t239)) / t243)) * 2.35_c_double) + (sin(((t29 * 1.43_c_double) + 2.8_c_double)) * 0.58_c_double)) - ((t120 * 0.3_c_double) / t127)) + ((t261 * 0.3_c_double) / t266)) + ((3.5_c_double - t16) * 0.08_c_double)) - (t22 * 0.54_c_double)) * t1))
    t446 = (t23 + (((((((((t5 * 0.0_c_double) + (sum(((t341 / t349) * t370)) / t374)) * 2.35_c_double) + (cos(((t29 * 1.79_c_double) + 4.2_c_double)) * 0.58_c_double)) - ((t121 * 0.3_c_double) / t132)) - ((t260 * 0.3_c_double) / t266)) + ((5.0_c_double - t17) * 0.08_c_double)) - (t23 * 0.54_c_double)) * t1))
    t448 = (t24 + (((((((((t6 * 0.0_c_double) + (sum(((t342 / t349) * t370)) / t374)) * 2.35_c_double) + (sin(((t29 * 1.31_c_double) + 5.0_c_double)) * 0.58_c_double)) - ((t122 * 0.3_c_double) / t132)) - ((t261 * 0.3_c_double) / t266)) + ((3.5_c_double - t18) * 0.08_c_double)) - (t24 * 0.54_c_double)) * t1))
    t198 = min(max((t13 + (t188 * t1)), 0.65_c_double), 9.35_c_double)
    t206 = min(max((t14 + (t190 * t1)), 0.65_c_double), 6.35_c_double)
    t332 = min(max((t15 + (t322 * t1)), 0.65_c_double), 9.35_c_double)
    t340 = min(max((t16 + (t324 * t1)), 0.65_c_double), 6.35_c_double)
    t456 = min(max((t17 + (t446 * t1)), 0.65_c_double), 9.35_c_double)
    t464 = min(max((t18 + (t448 * t1)), 0.65_c_double), 6.35_c_double)
    t492 = (((t5 - t198) * (t5 - t198)) + ((t6 - t206) * (t6 - t206)))
    t499 = (((t5 - t332) * (t5 - t332)) + ((t6 - t340) * (t6 - t340)))
    t506 = (((t5 - t456) * (t5 - t456)) + ((t6 - t464) * (t6 - t464)))
    t561 = max(max((min(max((t485 - abs((t5 - t198))), 0.0_c_double), max((t485 - abs((t6 - t206))), 0.0_c_double)) / t485), (min(max((t485 - abs((t5 - t332))), 0.0_c_double), max((t485 - abs((t6 - t340))), 0.0_c_double)) / t485)), (min(max((t485 - abs((t5 - t456))), 0.0_c_double), max((t485 - abs((t6 - t464))), 0.0_c_double)) / t485))
    t593 = min(((exp(((-t492) / 1.2168_c_double)) + exp(((-t499) / 1.2168_c_double))) + exp(((-t506) / 1.2168_c_double))), 1.0_c_double)
    t894 = (t561 * t561)
    t782 = min(((t7 * exp((t1 * -0.095_c_double))) + (((t1 * 3.2_c_double) * exp(((-t492) / 0.1152_c_double))) * t748)), 1.0_c_double)
    t796 = min(((t8 * exp((t1 * -0.072_c_double))) + (((t1 * 2.2_c_double) * exp(((-t492) / 0.2888_c_double))) * t748)), 1.0_c_double)
    t810 = min(((t9 * exp((t1 * -0.095_c_double))) + (((t1 * 3.2_c_double) * exp(((-t499) / 0.1152_c_double))) * t758)), 1.0_c_double)
    t824 = min(((t10 * exp((t1 * -0.072_c_double))) + (((t1 * 2.2_c_double) * exp(((-t499) / 0.2888_c_double))) * t758)), 1.0_c_double)
    t838 = min(((t11 * exp((t1 * -0.095_c_double))) + (((t1 * 3.2_c_double) * exp(((-t506) / 0.1152_c_double))) * t768)), 1.0_c_double)
    t852 = min(((t12 * exp((t1 * -0.072_c_double))) + (((t1 * 2.2_c_double) * exp(((-t506) / 0.2888_c_double))) * t768)), 1.0_c_double)
    t608 = (t27 + ((((((t593 * -0.42_c_double) - ((t561 * 0.22_c_double) * t561)) - t26) * 20.0_c_double) - (t27 * 8.0_c_double)) * t1))
    t860 = max((((((t782 + t796) + t810) + t824) + t838) + t852), 1.0e-06_c_double)
    t610 = (t26 + (t608 * t1))
    t863 = min(t860, 0.88_c_double)
    t630 = min(max(((-t610) / 0.42_c_double), 0.0_c_double), 1.0_c_double)
    t621 = min(max((((t28 + t610) - 0.5_c_double) / 5.0_c_double), 0.0_c_double), 1.0_c_double)
    t900 = ((((min(max(((((t621 * 27.0_c_double) + 186.0_c_double) - (t630 * 34.0_c_double)) + (t593 * 54.0_c_double)), 0.0_c_double), 255.0_c_double) * (1.0_c_double - t863)) + (((((t782 + t796) + t852) * 255.0_c_double) / t860) * t863)) * (1.0_c_double - t894)) + (t894 * 245.0_c_double))
    t906 = ((((min(max((((((t621 * 18.0_c_double) + 220.0_c_double) - (t630 * 21.0_c_double)) + (t593 * 30.0_c_double)) + (min(abs(t608), 1.0_c_double) * 8.0_c_double)), 0.0_c_double), 255.0_c_double) * (1.0_c_double - t863)) + (((((t796 + t810) + t824) * 255.0_c_double) / t860) * t863)) * (1.0_c_double - t894)) + (t894 * 252.0_c_double))
    t912 = ((((min(max(((((t621 * 16.0_c_double) + 232.0_c_double) + (t630 * 15.0_c_double)) + (t593 * 20.0_c_double)), 0.0_c_double), 255.0_c_double) * (1.0_c_double - t863)) + (((((t824 + t838) + t852) * 255.0_c_double) / t860) * t863)) * (1.0_c_double - t894)) + (t894 * 255.0_c_double))
    return
  end subroutine columnar_multifluid_rgb_step
  subroutine columnar_multifluid_rgb_step_control(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t900, t906, t912, t610, t608, t198, t206, t188, t190, t332, t340, t322, t324, t456, t464, t446, t448, t29, t782, t796, t810, t824, t838, t852) bind(C, name="columnar_multifluid_rgb_step_control")
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
    real(c_double), intent(out) :: t900(extent_16)
    real(c_double), intent(out) :: t906(extent_16)
    real(c_double), intent(out) :: t912(extent_16)
    real(c_double), intent(out) :: t610(extent_16)
    real(c_double), intent(out) :: t608(extent_16)
    real(c_double), intent(out) :: t198(extent_16)
    real(c_double), intent(out) :: t206(extent_16)
    real(c_double), intent(out) :: t188(extent_16)
    real(c_double), intent(out) :: t190(extent_16)
    real(c_double), intent(out) :: t332(extent_16)
    real(c_double), intent(out) :: t340(extent_16)
    real(c_double), intent(out) :: t322(extent_16)
    real(c_double), intent(out) :: t324(extent_16)
    real(c_double), intent(out) :: t456(extent_16)
    real(c_double), intent(out) :: t464(extent_16)
    real(c_double), intent(out) :: t446(extent_16)
    real(c_double), intent(out) :: t448(extent_16)
    real(c_double), intent(out) :: t29(extent_16)
    real(c_double), intent(out) :: t782(extent_16)
    real(c_double), intent(out) :: t796(extent_16)
    real(c_double), intent(out) :: t810(extent_16)
    real(c_double), intent(out) :: t824(extent_16)
    real(c_double), intent(out) :: t838(extent_16)
    real(c_double), intent(out) :: t852(extent_16)

    ! block entry
    call numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t188, t190, t322, t324, t446, t448, t198, t206, t332, t340, t456, t464, t782, t796, t810, t824, t838, t852, t608, t610, t900, t912, t906)
    return
  end subroutine columnar_multifluid_rgb_step_control
  subroutine numerical_region_0(extent_1, extent_16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t188, t190, t322, t324, t446, t448, t198, t206, t332, t340, t456, t464, t782, t796, t810, t824, t838, t852, t608, t610, t900, t912, t906) bind(C, name="numerical_region_0")
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
    real(c_double), intent(out) :: t29(extent_16)
    real(c_double), intent(out) :: t188(extent_16)
    real(c_double), intent(out) :: t190(extent_16)
    real(c_double), intent(out) :: t322(extent_16)
    real(c_double), intent(out) :: t324(extent_16)
    real(c_double), intent(out) :: t446(extent_16)
    real(c_double), intent(out) :: t448(extent_16)
    real(c_double), intent(out) :: t198(extent_16)
    real(c_double), intent(out) :: t206(extent_16)
    real(c_double), intent(out) :: t332(extent_16)
    real(c_double), intent(out) :: t340(extent_16)
    real(c_double), intent(out) :: t456(extent_16)
    real(c_double), intent(out) :: t464(extent_16)
    real(c_double), intent(out) :: t782(extent_16)
    real(c_double), intent(out) :: t796(extent_16)
    real(c_double), intent(out) :: t810(extent_16)
    real(c_double), intent(out) :: t824(extent_16)
    real(c_double), intent(out) :: t838(extent_16)
    real(c_double), intent(out) :: t852(extent_16)
    real(c_double), intent(out) :: t608(extent_16)
    real(c_double), intent(out) :: t610(extent_16)
    real(c_double), intent(out) :: t900(extent_16)
    real(c_double), intent(out) :: t912(extent_16)
    real(c_double), intent(out) :: t906(extent_16)
    real(c_double) :: t30(extent_16)
    real(c_double) :: t31(extent_1)
    real(c_double) :: t44(extent_1)
    real(c_double) :: t46(extent_1)
    real(c_double) :: t49(extent_1)
    real(c_double) :: t51(extent_1)
    real(c_double) :: t68(extent_16)
    real(c_double) :: t67(extent_16)
    real(c_double) :: t70(extent_16)
    real(c_double) :: t69(extent_16)
    real(c_double) :: t66(extent_16)
    real(c_double) :: t107(extent_1)
    real(c_double) :: t119(extent_16)
    real(c_double) :: t122(extent_16)
    real(c_double) :: t120(extent_16)
    real(c_double) :: t121(extent_16)
    real(c_double) :: t153(extent_1)
    real(c_double) :: t157(extent_1)
    real(c_double) :: t180(extent_1)
    real(c_double) :: t208(extent_16)
    real(c_double) :: t207(extent_16)
    real(c_double) :: t260(extent_16)
    real(c_double) :: t261(extent_16)
    real(c_double) :: t341(extent_16)
    real(c_double) :: t342(extent_16)
    real(c_double) :: t465(extent_1)
    real(c_double) :: t467(extent_1)
    real(c_double) :: t95(extent_1)
    real(c_double) :: t473(extent_1)
    real(c_double) :: t604(extent_1)
    real(c_double) :: t770(extent_1)
    real(c_double) :: t775(extent_1)
    real(c_double) :: t784(extent_1)
    real(c_double) :: t789(extent_1)
    real(c_double) :: t798(extent_1)
    real(c_double) :: t812(extent_1)
    real(c_double) :: t826(extent_1)
    real(c_double) :: t840(extent_1)
    real(c_double) :: t90(extent_1)
    real(c_double) :: t92(extent_1)
    real(c_double) :: t136(extent_1)
    real(c_double) :: t74(extent_1)
    real(c_double) :: t163(extent_1)
    real(c_double) :: t270(extent_1)
    real(c_double) :: t297(extent_1)
    real(c_double) :: t394(extent_1)
    real(c_double) :: t421(extent_1)
    real(c_double) :: t743(extent_1)
    real(c_double) :: t751(extent_1)
    real(c_double) :: t761(extent_1)
    real(c_double) :: t34(extent_16)
    real(c_double) :: t73(extent_16)
    real(c_double) :: t138(extent_1)
    real(c_double) :: t165(extent_1)
    real(c_double) :: t211(extent_16)
    real(c_double) :: t272(extent_1)
    real(c_double) :: t299(extent_1)
    real(c_double) :: t345(extent_16)
    real(c_double) :: t396(extent_1)
    real(c_double) :: t753(extent_1)
    real(c_double) :: t56(extent_1)
    real(c_double) :: t87(extent_1)
    real(c_double) :: t127(extent_16)
    real(c_double) :: t132(extent_16)
    real(c_double) :: t266(extent_16)
    real(c_double) :: t597(extent_1)
    real(c_double) :: t58(extent_16)
    real(c_double) :: t77(extent_16)
    real(c_double) :: t83(extent_1)
    real(c_double) :: t135(extent_1)
    real(c_double) :: t215(extent_16)
    real(c_double) :: t221(extent_1)
    real(c_double) :: t349(extent_16)
    real(c_double) :: t355(extent_1)
    real(c_double) :: t564(extent_1)
    real(c_double) :: t748(extent_16)
    real(c_double) :: t38(extent_1)
    real(c_double) :: t60(extent_16)
    real(c_double) :: t62(extent_16)
    real(c_double) :: t480(extent_1)
    real(c_double) :: t758(extent_16)
    real(c_double) :: t227(extent_16)
    real(c_double) :: t768(extent_16)
    real(c_double) :: t89(extent_16)
    real(c_double) :: t41(extent_16)
    real(c_double) :: t361(extent_16)
    real(c_double) :: t483(extent_1)
    real(c_double) :: t42(extent_16)
    real(c_double) :: t43(extent_16)
    real(c_double) :: t485(extent_16)
    real(c_double) :: t101(extent_1)
    real(c_double) :: t102
    real(c_double) :: t243
    real(c_double) :: t374
    real(c_double) :: t65(extent_16)
    real(c_double) :: t239(extent_16)
    real(c_double) :: t98(extent_16)
    real(c_double) :: t370(extent_16)
    real(c_double) :: t133(extent_1)
    real(c_double) :: t194(extent_1)
    real(c_double) :: t197(extent_1)
    real(c_double) :: t205(extent_1)
    real(c_double) :: t492(extent_16)
    real(c_double) :: t499(extent_16)
    real(c_double) :: t506(extent_16)
    real(c_double) :: t567(extent_1)
    real(c_double) :: t576(extent_1)
    real(c_double) :: t585(extent_1)
    real(c_double) :: t692(extent_1)
    real(c_double) :: t701(extent_1)
    real(c_double) :: t710(extent_1)
    real(c_double) :: t719(extent_1)
    real(c_double) :: t728(extent_1)
    real(c_double) :: t737(extent_1)
    real(c_double) :: t561(extent_16)
    real(c_double) :: t593(extent_16)
    real(c_double) :: t894(extent_16)
    real(c_double) :: t595(extent_1)
    real(c_double) :: t643(extent_1)
    real(c_double) :: t659(extent_1)
    real(c_double) :: t601(extent_1)
    real(c_double) :: t898(extent_1)
    real(c_double) :: t904(extent_1)
    real(c_double) :: t650(extent_1)
    real(c_double) :: t860(extent_16)
    real(c_double) :: t862(extent_1)
    real(c_double) :: t863(extent_16)
    real(c_double) :: t612(extent_1)
    real(c_double) :: t630(extent_16)
    real(c_double) :: t621(extent_16)
    real(c_double) :: t640(extent_1)
    real(c_double) :: t656(extent_1)
    real(c_double) :: t675(extent_1)
    real(c_double) :: t637(extent_1)
    real(c_double) :: t653(extent_1)
    real(c_double) :: t672(extent_1)
    real(c_double) :: t636(extent_1)
    real(c_double) :: t652(extent_1)
    real(c_double) :: t671(extent_1)

    ! block entry
    t30 = (t2 - t3)
    t29 = (t0 + t1)
    t31 = 2.0_c_double
    t44 = 0.61_c_double
    t46 = 0.83_c_double
    t49 = 0.37_c_double
    t51 = 0.29_c_double
    t68 = (t11 + t12)
    t67 = (t9 + t10)
    t70 = (t6 - t14)
    t69 = (t5 - t13)
    t66 = (t7 + t8)
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t119 = (t13 - t15)
    t122 = (t14 - t18)
    t120 = (t14 - t16)
    t121 = (t13 - t17)
    t153 = 5.0_c_double
    t157 = 0.54_c_double
    t180 = 3.5_c_double
    t157 = 0.54_c_double
    t208 = (t6 - t16)
    t207 = (t5 - t15)
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t260 = (t15 - t17)
    t261 = (t16 - t18)
    t153 = 5.0_c_double
    t157 = 0.54_c_double
    t180 = 3.5_c_double
    t157 = 0.54_c_double
    t341 = (t5 - t17)
    t342 = (t6 - t18)
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t153 = 5.0_c_double
    t157 = 0.54_c_double
    t180 = 3.5_c_double
    t157 = 0.54_c_double
    t465 = 0.42_c_double
    t467 = 0.34_c_double
    t95 = 0.18_c_double
    t473 = 0.35_c_double
    t604 = 8.0_c_double
    t770 = -0.095_c_double
    t775 = 3.2_c_double
    t784 = -0.072_c_double
    t789 = 2.2_c_double
    t798 = -0.095_c_double
    t775 = 3.2_c_double
    t812 = -0.072_c_double
    t789 = 2.2_c_double
    t826 = -0.095_c_double
    t775 = 3.2_c_double
    t840 = -0.072_c_double
    t789 = 2.2_c_double
    t90 = 1.35_c_double
    t92 = 0.3_c_double
    t136 = 1.71_c_double
    t92 = 0.3_c_double
    t92 = 0.3_c_double
    t74 = 0.08_c_double
    t163 = 1.37_c_double
    t92 = 0.3_c_double
    t92 = 0.3_c_double
    t74 = 0.08_c_double
    t90 = 1.35_c_double
    t92 = 0.3_c_double
    t270 = 1.63_c_double
    t92 = 0.3_c_double
    t92 = 0.3_c_double
    t74 = 0.08_c_double
    t297 = 1.43_c_double
    t92 = 0.3_c_double
    t92 = 0.3_c_double
    t74 = 0.08_c_double
    t90 = 1.35_c_double
    t92 = 0.3_c_double
    t394 = 1.79_c_double
    t92 = 0.3_c_double
    t92 = 0.3_c_double
    t74 = 0.08_c_double
    t421 = 1.31_c_double
    t92 = 0.3_c_double
    t92 = 0.3_c_double
    t74 = 0.08_c_double
    t743 = 2.11_c_double
    t751 = 1.91_c_double
    t761 = 2.27_c_double
    t34 = (((t4 * t31(1)) - t2) - t3)
    t73 = ((t69 * t69) + (t70 * t70))
    t138 = 0.2_c_double
    t165 = 1.1_c_double
    t211 = ((t207 * t207) + (t208 * t208))
    t272 = 2.3_c_double
    t299 = 2.8_c_double
    t345 = ((t341 * t341) + (t342 * t342))
    t396 = 4.2_c_double
    t153 = 5.0_c_double
    t753 = 2.1_c_double
    t396 = 4.2_c_double
    t56 = 0.72_c_double
    t74 = 0.08_c_double
    t87 = 0.14_c_double
    t95 = 0.18_c_double
    t127 = (((t119 * t119) + (t120 * t120)) + t95(1))
    t95 = 0.18_c_double
    t132 = (((t121 * t121) + (t122 * t122)) + t95(1))
    t74 = 0.08_c_double
    t87 = 0.14_c_double
    t95 = 0.18_c_double
    t266 = (((t260 * t260) + (t261 * t261)) + t95(1))
    t74 = 0.08_c_double
    t87 = 0.14_c_double
    t597 = 0.22_c_double
    t58 = (((t5 * t44(1)) + (t6 * t46(1))) + (sin(((t5 * t49(1)) - (t6 * t51(1)))) * t56(1)))
    t77 = sqrt((t73 + t74(1)))
    t83 = 4.805000000000001_c_double
    t135 = 0.58_c_double
    t135 = 0.58_c_double
    t215 = sqrt((t211 + t74(1)))
    t221 = 4.805000000000001_c_double
    t135 = 0.58_c_double
    t135 = 0.58_c_double
    t349 = sqrt((t345 + t74(1)))
    t355 = 4.805000000000001_c_double
    t135 = 0.58_c_double
    t135 = 0.58_c_double
    t107 = 0.0_c_double
    t564 = 0.78_c_double
    t748 = ((sin((t29 * t743(1))) * t597(1)) + t564(1))
    t597 = 0.22_c_double
    t597 = 0.22_c_double
    t38 = 1.0e-05_c_double
    t60 = cos(t58)
    t62 = sin(t58)
    t480 = 1.0_c_double
    t564 = 0.78_c_double
    t758 = ((sin(((t29 * t751(1)) + t753(1))) * t597(1)) + t564(1))
    t564 = 0.78_c_double
    t227 = (exp(((-t211) / t221(1))) / (t211 + t87(1)))
    t768 = ((sin(((t29 * t761(1)) + t396(1))) * t597(1)) + t564(1))
    t89 = (exp(((-t73) / t83(1))) / (t73 + t87(1)))
    t41 = sqrt((((t30 * t30) + (t34 * t34)) + t38(1)))
    t361 = (exp(((-t345) / t355(1))) / (t345 + t87(1)))
    t483 = 0.12_c_double
    t42 = (t30 / t41)
    t43 = (t34 / t41)
    t95 = 0.18_c_double
    t485 = ((min(max(((((t2 * t465(1)) + (t4 * t467(1))) + (t3 * t95(1))) + (t25 * t473(1))), t107(1)), t480(1)) * t483(1)) + t95(1))
    t101 = 1.0e-06_c_double
    t102 = (sum(t89) + t101(1))
    t101 = 1.0e-06_c_double
    t243 = (sum(t227) + t101(1))
    t101 = 1.0e-06_c_double
    t374 = (sum(t361) + t101(1))
    t65 = ((t60 * t42) + (t62 * t43))
    t95 = 0.18_c_double
    t95 = 0.18_c_double
    t95 = 0.18_c_double
    t239 = (t227 * (((t66 * t90(1)) - (t67 * t92(1))) + (((t62 * t42) - (t60 * t43)) * t95(1))))
    t98 = (t89 * (((t68 * t90(1)) - (t66 * t92(1))) + (t65 * t95(1))))
    t370 = (t361 * (((t67 * t90(1)) - (t68 * t92(1))) - (t65 * t95(1))))
    t133 = 2.35_c_double
    t133 = 2.35_c_double
    t133 = 2.35_c_double
    t133 = 2.35_c_double
    t133 = 2.35_c_double
    t133 = 2.35_c_double
    t188 = (t19 + (((((((((t5 * t107(1)) + (sum(((t69 / t77) * t98)) / t102)) * t133(1)) + (cos(((t29 * t136(1)) + t138(1))) * t135(1))) + ((t119 * t92(1)) / t127)) + ((t121 * t92(1)) / t132)) + ((t153(1) - t13) * t74(1))) - (t19 * t157(1))) * t1))
    t190 = (t20 + (((((((((t6 * t107(1)) + (sum(((t70 / t77) * t98)) / t102)) * t133(1)) + (sin(((t29 * t163(1)) + t165(1))) * t135(1))) + ((t120 * t92(1)) / t127)) + ((t122 * t92(1)) / t132)) + ((t180(1) - t14) * t74(1))) - (t20 * t157(1))) * t1))
    t322 = (t21 + (((((((((t5 * t107(1)) + (sum(((t207 / t215) * t239)) / t243)) * t133(1)) + (cos(((t29 * t270(1)) + t272(1))) * t135(1))) - ((t119 * t92(1)) / t127)) + ((t260 * t92(1)) / t266)) + ((t153(1) - t15) * t74(1))) - (t21 * t157(1))) * t1))
    t324 = (t22 + (((((((((t6 * t107(1)) + (sum(((t208 / t215) * t239)) / t243)) * t133(1)) + (sin(((t29 * t297(1)) + t299(1))) * t135(1))) - ((t120 * t92(1)) / t127)) + ((t261 * t92(1)) / t266)) + ((t180(1) - t16) * t74(1))) - (t22 * t157(1))) * t1))
    t446 = (t23 + (((((((((t5 * t107(1)) + (sum(((t341 / t349) * t370)) / t374)) * t133(1)) + (cos(((t29 * t394(1)) + t396(1))) * t135(1))) - ((t121 * t92(1)) / t132)) - ((t260 * t92(1)) / t266)) + ((t153(1) - t17) * t74(1))) - (t23 * t157(1))) * t1))
    t448 = (t24 + (((((((((t6 * t107(1)) + (sum(((t342 / t349) * t370)) / t374)) * t133(1)) + (sin(((t29 * t421(1)) + t153(1))) * t135(1))) - ((t122 * t92(1)) / t132)) - ((t261 * t92(1)) / t266)) + ((t180(1) - t18) * t74(1))) - (t24 * t157(1))) * t1))
    t194 = 0.65_c_double
    t194 = 0.65_c_double
    t194 = 0.65_c_double
    t194 = 0.65_c_double
    t194 = 0.65_c_double
    t194 = 0.65_c_double
    t197 = 9.35_c_double
    t198 = min(max((t13 + (t188 * t1)), t194(1)), t197(1))
    t205 = 6.35_c_double
    t206 = min(max((t14 + (t190 * t1)), t194(1)), t205(1))
    t197 = 9.35_c_double
    t332 = min(max((t15 + (t322 * t1)), t194(1)), t197(1))
    t205 = 6.35_c_double
    t340 = min(max((t16 + (t324 * t1)), t194(1)), t205(1))
    t197 = 9.35_c_double
    t456 = min(max((t17 + (t446 * t1)), t194(1)), t197(1))
    t205 = 6.35_c_double
    t464 = min(max((t18 + (t448 * t1)), t194(1)), t205(1))
    t492 = (((t5 - t198) * (t5 - t198)) + ((t6 - t206) * (t6 - t206)))
    t499 = (((t5 - t332) * (t5 - t332)) + ((t6 - t340) * (t6 - t340)))
    t506 = (((t5 - t456) * (t5 - t456)) + ((t6 - t464) * (t6 - t464)))
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t567 = 1.2168_c_double
    t576 = 1.2168_c_double
    t585 = 1.2168_c_double
    t692 = 0.1152_c_double
    t701 = 0.2888_c_double
    t710 = 0.1152_c_double
    t719 = 0.2888_c_double
    t728 = 0.1152_c_double
    t737 = 0.2888_c_double
    t561 = max(max((min(max((t485 - abs((t5 - t198))), t107(1)), max((t485 - abs((t6 - t206))), t107(1))) / t485), (min(max((t485 - abs((t5 - t332))), t107(1)), max((t485 - abs((t6 - t340))), t107(1))) / t485)), (min(max((t485 - abs((t5 - t456))), t107(1)), max((t485 - abs((t6 - t464))), t107(1))) / t485))
    t480 = 1.0_c_double
    t593 = min(((exp(((-t492) / t567(1))) + exp(((-t499) / t576(1)))) + exp(((-t506) / t585(1)))), t480(1))
    t597 = 0.22_c_double
    t894 = (t561 * t561)
    t595 = -0.42_c_double
    t643 = 54.0_c_double
    t659 = 30.0_c_double
    t601 = 20.0_c_double
    t480 = 1.0_c_double
    t782 = min(((t7 * exp((t1 * t770(1)))) + (((t1 * t775(1)) * exp(((-t492) / t692(1)))) * t748)), t480(1))
    t480 = 1.0_c_double
    t796 = min(((t8 * exp((t1 * t784(1)))) + (((t1 * t789(1)) * exp(((-t492) / t701(1)))) * t748)), t480(1))
    t480 = 1.0_c_double
    t810 = min(((t9 * exp((t1 * t798(1)))) + (((t1 * t775(1)) * exp(((-t499) / t710(1)))) * t758)), t480(1))
    t480 = 1.0_c_double
    t824 = min(((t10 * exp((t1 * t812(1)))) + (((t1 * t789(1)) * exp(((-t499) / t719(1)))) * t758)), t480(1))
    t480 = 1.0_c_double
    t838 = min(((t11 * exp((t1 * t826(1)))) + (((t1 * t775(1)) * exp(((-t506) / t728(1)))) * t768)), t480(1))
    t480 = 1.0_c_double
    t852 = min(((t12 * exp((t1 * t840(1)))) + (((t1 * t789(1)) * exp(((-t506) / t737(1)))) * t768)), t480(1))
    t480 = 1.0_c_double
    t898 = 245.0_c_double
    t480 = 1.0_c_double
    t904 = 252.0_c_double
    t480 = 1.0_c_double
    t650 = 255.0_c_double
    t601 = 20.0_c_double
    t650 = 255.0_c_double
    t650 = 255.0_c_double
    t650 = 255.0_c_double
    t608 = (t27 + ((((((t593 * t595(1)) - ((t561 * t597(1)) * t561)) - t26) * t601(1)) - (t27 * t604(1))) * t1))
    t101 = 1.0e-06_c_double
    t860 = max((((((t782 + t796) + t810) + t824) + t838) + t852), t101(1))
    t862 = 0.88_c_double
    t610 = (t26 + (t608 * t1))
    t863 = min(t860, t862(1))
    t480 = 1.0_c_double
    t480 = 1.0_c_double
    t480 = 1.0_c_double
    t480 = 1.0_c_double
    t604 = 8.0_c_double
    t612 = 0.5_c_double
    t465 = 0.42_c_double
    t153 = 5.0_c_double
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t480 = 1.0_c_double
    t630 = min(max(((-t610) / t465(1)), t107(1)), t480(1))
    t480 = 1.0_c_double
    t621 = min(max((((t28 + t610) - t612(1)) / t153(1)), t107(1)), t480(1))
    t640 = 34.0_c_double
    t656 = 21.0_c_double
    t675 = 15.0_c_double
    t637 = 27.0_c_double
    t653 = 18.0_c_double
    t672 = 16.0_c_double
    t636 = 186.0_c_double
    t652 = 220.0_c_double
    t671 = 232.0_c_double
    t107 = 0.0_c_double
    t107 = 0.0_c_double
    t650 = 255.0_c_double
    t107 = 0.0_c_double
    t650 = 255.0_c_double
    t650 = 255.0_c_double
    t900 = ((((min(max(((((t621 * t637(1)) + t636(1)) - (t630 * t640(1))) + (t593 * t643(1))), t107(1)), t650(1)) * (t480(1) - t863)) + (((((t782 + t796) + t852) * t650(1)) / t860) * t863)) * (t480(1) - t894)) + (t894 * t898(1)))
    t906 = ((((min(max((((((t621 * t653(1)) + t652(1)) - (t630 * t656(1))) + (t593 * t659(1))) + (min(abs(t608), t480(1)) * t604(1))), t107(1)), t650(1)) * (t480(1) - t863)) + (((((t796 + t810) + t824) * t650(1)) / t860) * t863)) * (t480(1) - t894)) + (t894 * t904(1)))
    t912 = ((((min(max(((((t621 * t672(1)) + t671(1)) + (t630 * t675(1))) + (t593 * t601(1))), t107(1)), t650(1)) * (t480(1) - t863)) + (((((t824 + t838) + t852) * t650(1)) / t860) * t863)) * (t480(1) - t894)) + (t894 * t650(1)))
    return
  end subroutine numerical_region_0

end module kernel_fortran
