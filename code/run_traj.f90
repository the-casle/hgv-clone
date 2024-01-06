program sim
    implicit none
    call run_traj ( )
end program sim


subroutine run_traj ( )
  
     
      use COESA_module 
      !  This is an atmospheric model we can replace with exponetial density model
      !  rho = rho_0 exp( -Beta h). Where Beta = 0.1378 or g/RT for earth.

      use NA
      !The Newtonian aero model for use
      implicit none
      Character(50) :: fname,fresults ! The input and output file names
      integer ( kind = 4 ), parameter :: neqn = 5 ! we have 4 ODE's and 1 dummy variable for angle of attack
  
      real ( kind = 8 ) abserr, rho,P_inf,T_inf
      integer ( kind = 4 ) flag
      integer ( kind = 4 ) err
      external EOM ! using external equation which houses equation of motion (EOM)

      ! define variables need for time integration
      real ( kind = 8 ) relerr
      real ( kind = 8 ) t
      real ( kind = 8 ) t1
      real ( kind = 8 ) dt
      real ( kind = 8 )  q
      real ( kind = 8 )  qt
      ! allocate 8 bytes, this is what kind = 8 mean or type real arrays of size neqn. allocate and array of doubles
      real ( kind = 8 ) y(neqn)  ! state vector, y
      real ( kind = 8 ) yp(neqn)  ! derivate of state vector. y'
      ntri= 187140!417420!187140!  !needs to be dynamically changed. 
      fname="./geometry/glider_187k.stl"!"./geometry/glider_187k.stl" ! name of input file
      fresults="./data/trajNA187.txt" ! output file name
      !417420!187140!  !needs to be dynamically changed. 
      call read_STL(fname)        ! read in the stl file and store data needed to compute L/D. 
      open(unit=9,file=fresults,status="replace", position="append", action="write")

      ! Have some default values for integration and starting values for simulation. 
      abserr = 1d-9
      relerr = 1d-9
      flag = 1
      t = 0.0D+00
      dt = 2d0! usally 10 s for RK4 else 0.1 for standard euler. 
      t1=dt+t
      y = (/ 0.d0,6d3,76.2d3,0d0,3.5d-1/) ! gamma (0 rads), vel (6 km/s), altitude (60 km), downrange (0 km), alpha (0.35 rads)
      err=COESA_density(y(3), rho,P_inf,T_inf) ! initiliaze the correct atmospheric values. 
   
      call EOM ( t, y, yp ) ! initiliaze the yp or y' or dy/dt vector
      write(9,*) t,y,L_D,rho, 0.5 * rho*y(2)**2 ! write data for output file 

      do 
          IF ((y(2).le.1d3) .or. (y(3).le.1d3) .or. (y(3).gt.80d3)) EXIT 
          ! If the altitude y(3) is less than 1km exit the loop/ stop the simulation
          ! If the altitude y(3) is greater than 80km exit the loop/ stop the simulation. You are about to be in space my guy. 
          ! If the speed y(2) is less than 1km/s exit the loop/ stop the simulation

          call r8_rkf45 ( EOM, neqn, y, yp, t, t1, relerr, abserr, flag )
          ! preform time integration using Runga-Kutta with 4 stages. 
          ! else we can do a neive approach 
          ! y=y+yp*dt->
          ! call EOM(t,y,yp) to get new derivative values for next time step

          err=COESA_density(y(3), rho,P_inf,T_inf) ! get atmospheric values at new height
          write(9,*) t,y,L_D,rho, 0.5 * rho*y(2)**2 ! write data for output file 
          ! update the ttme t
          t=t1 
          t1=t1+dt
          
      end do
  
      return ! return nothing
      end
  
  
 
    subroutine compute_g(h, g) 
        ! function use to compute variation in gravity
  
      implicit none
      real ( kind = 8 ), intent(in):: h
      real ( kind = 8 ), intent(out):: g
      real ( kind = 8 ):: R 
      real ( kind = 8 ) :: g_e 
      R= 6.371d6 ! radius of the earth
      g_e = 9.80665d0 ! sea lvl gravity accel
  
      g= g_e*(R/(R+h))**2
  
    return 
    end
  
   subroutine EOM ( t, y, yp )
    ! equations of motion
      use COESA_module
      use NA
      implicit none
      integer ( kind = 4 ) err
      real ( kind = 8 ) t,T_inf,L,D
      real ( kind = 8 ) rho
      real ( kind = 8 ) y(5)
      real ( kind = 8 ) yp(5)
      external compute_g    
      real ( kind = 8 ) gamma
      real ( kind = 8 ) v
      real ( kind = 8 ) alpha
      real ( kind = 8 ) h
      real ( kind = 8 ) x 
      ! real ( kind = 8 ) b ! ballistic coefficent. 
      real ( kind = 8 ) g
      ! real ( kind = 8 ) L_D 
      ! real ( kind = 8 ) Cd
      real ( kind = 8 ) mass
      real ( kind = 8 ):: R 
      real ( kind = 8 ):: P_inf
      ! type(State) :: res
      R= 6.371d6 ! radius of the earth in Km
      mass= 1.0d3! assumed mass is fixed
      ! L=3.67d0
      ! parameters to be computed by CFD
      gamma=y(1)
      v=y(2)
      h=y(3)
      x=y(4) 
      alpha = y(5) 
  
      err=COESA_density(h, rho,P_inf,T_inf) ! return rho, and P_inf
      call calc_LD(alpha,v,rho,P_inf, L,D)  ! computes the cd and L_D where L_D is a global variable from NA
      call compute_g(h,g)
      ! b =  mass/(ref_area*Cd)
  
      yp(1) =(1/v) *(-L/mass+g*cos(gamma)-(v**2/(R+h))*cos(gamma)) ! flight path angle. 
      yp(2) = -D/mass+g*sin(gamma) ! Velocity (truely air speed).
      yp(3)= -v*sin(gamma) ! height in meters
      yp(4) = v*cos(gamma) ! cross range distance.
      yp(5) =0d0 !(cos(t)*(t+2)-sin(t))/(t+2)**2 ! alpha doesn't change here
      ! write ( *, *) yp
  
      return
      end
  
  