!-----------------------------------------------------
! Produces Mandelbrot plot in the range [-ext[0]:ext[1]]x[ext[2]:ext[3]]
! It uses formula z = z*z + z0 iteratively until
! asb(z) gets bigger than 2 
! (deciding that z0 is not in mandelbrot)
! The value returned is 1/(#-iterations to escape)
!-----------------------------------------------------
SUBROUTINE Mandelb(data, ext, Nx, Ny, max_steps)
  IMPLICIT NONE ! Don't use any implicit names of variables!
  ! Function arguments
  REAL*8, intent(out) :: data(Nx,Ny)
  REAL*8, intent(in)  :: ext(4)              ! [xa,xb,ya,yb]
  INTEGER, intent(in) :: max_steps
  INTEGER, intent(in) :: Nx, Ny
  !f2py integer optional, intent(in)          :: max_steps=1000
  !  !f2py integer intent(hide), depend(data) :: Nx=shape(data,0)  ! it will be hidden automatically
  !  !f2py integer intent(hide), depend(data) :: Ny=shape(data,1)  ! it will be hidden automatically
  ! Local variables
  INTEGER    :: i, j, itt
  COMPLEX*16 :: z0, z
  REAL*8     :: x, y
  data(:,:) = max_steps
  !$OMP PARALLEL DO  PRIVATE(j,x,y,z0,z,itt)
  DO i=1,Nx
     DO j=1,Ny
        x = ext(1) + (ext(2)-ext(1))*(i-1.)/(Nx-1.)
        y = ext(3) + (ext(4)-ext(3))*(j-1.)/(Ny-1.)
        z0 = dcmplx(x,y)
        z=0
        DO itt=1,max_steps
           IF (abs(z)>2.) THEN
              data(i,j) = itt-1 !1./itt         ! result is number of iterations
              EXIT
           ENDIF
           z = z**2 + z0             ! f(z) = z**2+z0 -> z
        ENDDO
        !if (abs(z)<2) data(i,j) = max_steps
     ENDDO
  ENDDO
  !$OMP END PARALLEL DO
  RETURN
END SUBROUTINE Mandelb
