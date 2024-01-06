!  ***********************************************************************************
      module NA
      implicit none

      integer  ntri ! hard coded to be this number of triangles. 
      real (kind =8), allocatable, dimension(:) :: area
      real (kind =8), allocatable, dimension(:,:) :: normals
       real (kind =8) L_D, ref_area
      PUBLIC :: ntri, ref_area,L_D
      PRIVATE :: normals,area
      save
 
    CONTAINS

! ---------------------------------------------------------
subroutine read_STL(fname)
    Implicit none
    
    Character(LEN=*), intent(IN) :: fname
    Character(10) :: void_chars
    integer ::  iunit,k,i
    real (kind=8) :: vec1(3),vec2(3), area_(3),r1(3),r2(3),r3(3)

    ! ntri= 187140!417420!187140!  !needs to be dynamically changed. 
    ! ntri=19602 ! for flat plate
    iunit=9
    allocate(area(ntri))
    allocate(normals(ntri,3))
    ! allocate(centers(ntri,3))
    ref_area=0.0d0
    ! centers=(/ 1.8116579d0, 0.36715084d0, 0.0d0 /)
    OPEN(unit=iunit,file=fname)
    read(iunit,*) ! skip line 1
    
    do i = 1,ntri
        READ(iunit,*) void_chars,void_chars, normals(i,:)
        READ(iunit,*) ! skip non-numbers 

        READ(iunit,*) void_chars,r1(1),r1(2),r1(3)
        READ(iunit,*) void_chars,r2(1),r2(2),r2(3)
        READ(iunit,*) void_chars,r3(1),r3(2),r3(3)

        READ(iunit,*) ! skip non-numbers 
        READ(iunit,*) ! skip non-numbers 
        ! compute the areas
        ! centers(i,:)=(r1+r2+r3)/2
        vec1 = r1(:)-r3(:)
        vec2 = r1(:)-r2(:)
        area_ = cross(vec1,vec2)
        area(i) = 0.5*NORM2(area_)
        ! compute refrence area
        ref_area = ref_area + area(i)
        !TODO delte normalize norms
        ! nlen = NORM2(normals(i,:))
        ! normals(i,:)=normals(i,:)/nlen
       
    end do  
end subroutine read_STL

subroutine calc_LD(alpha,vel,rho,P_inf, L,D)
    Implicit none
    real (kind=8), intent(IN) :: alpha, vel, rho,P_inf
    real (kind=8), intent(out) :: L,D
    real (kind=8)  u(3), F_Total(3)
    real (kind=8) P,q,vel_,cp
    integer  i

    q = 0.5 * rho*vel**2
    u(1) = COS(alpha)
    u(2) = SIN(alpha)
    u(3)=0.0d0
    F_Total= (/ 0.0, 0.0, 0.0 /)
    
    do i = 1,ntri
        vel_ = DOT_PRODUCT(u,normals(i,:))
        IF ( vel_ .lt. 0.0) then 
            cp = 2*vel_**2       
        ELSE
            cp=0
        end IF
        P=(cp*q)+P_inf 
        F_Total = F_Total + (P*area(i))*(-normals(i,:))
        
    end do
    D=F_total(2)*SIN(alpha)+F_total(1)*COS(alpha);
    L=F_total(2)*COS(alpha)-F_total(1)*SIN(alpha);
    ! write(*,*) F_total(:)
    L_D = L/D 
    ! Cd = D/(q*ref_area)
    ! Cl = L/(q*ref_area)
end subroutine calc_LD
FUNCTION cross(a, b)
  real (kind=8) ::  cross(3)
  real (kind=8), INTENT(IN) :: a(3), b(3)

  cross(1) = a(2) * b(3) - a(3) * b(2)
  cross(2) = a(3) * b(1) - a(1) * b(3)
  cross(3) = a(1) * b(2) - a(2) * b(1)
END FUNCTION cross

end module NA

