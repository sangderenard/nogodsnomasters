module kernel_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains

  subroutine render(extent_4, t2725003784144, t2725003787152, t2725003787536, t2725023979984) bind(C, name="render")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_4
    real(c_double), intent(in) :: t2725003784144(extent_4)
    real(c_double), intent(in) :: t2725003787152(extent_4)
    real(c_double), intent(in) :: t2725003787536(extent_4)
    real(c_double), intent(out) :: t2725023979984(extent_4)

    ! block entry
    t2725023979984 = (((t2725003787536 * t2725003787152) + (t2725003784144 * (1.0_c_double - t2725003787152))) * 255.0_c_double)
    return
  end subroutine render
  subroutine render_control(extent_4, t0, t1, t2) bind(C, name="render_control")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_4
    real(c_double), intent(in) :: t0(extent_4)
    real(c_double), intent(in) :: t1(extent_4)
    real(c_double), intent(in) :: t2(extent_4)
    real(c_double) :: t9(extent_4)

    ! block entry
    call numerical_region_0(extent_4, t0, t1, t2, t9)
    return
  end subroutine render_control
  subroutine numerical_region_0(extent_4, t0, t1, t2, t9) bind(C, name="numerical_region_0")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: extent_4
    real(c_double), intent(in) :: t0(extent_4)
    real(c_double), intent(in) :: t1(extent_4)
    real(c_double), intent(in) :: t2(extent_4)
    real(c_double), intent(out) :: t9(extent_4)

    ! block entry
    t9 = (((t0 * t1) + (t2 * (1.0_c_double - t1))) * 255.0_c_double)
    return
  end subroutine numerical_region_0

end module kernel_fortran
