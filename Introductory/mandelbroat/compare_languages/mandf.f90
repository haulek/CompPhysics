INTEGER Function Mandelb(z0, max_steps)
  IMPLICIT NONE ! Every variable needs to be declared. It is very prudent to use that.
  COMPLEX*16, intent(in) :: z0
  INTEGER, intent(in)    :: max_steps
  ! locals
  COMPLEX*16 :: z
  INTEGER    :: i
  z=0.
  do i=1,max_steps
     if (abs(z)>2.) then
        Mandelb = i-1
        return 
     end if
     z = z*z + z0
  end do
  Mandelb = max_steps
  return
END Function Mandelb

program mand
  use omp_lib
  IMPLICIT NONE
  ! external function
  INTEGER :: Mandelb ! Need to declare the external function
  ! locals
  INTEGER :: i, j
  REAL*8  :: x, y
  COMPLEX*16  :: z0
  INTEGER, parameter :: Nx = 1000
  INTEGER, parameter :: Ny = 1000
  INTEGER, parameter :: max_steps = 1000
  REAL*8  :: ext(4) = (/-2., 1., -1., 1./) ! The limits of plotting
  REAL*8  :: mande(Nx,Ny)
  REAL    :: start, finish, startw, finishw
  
  call cpu_time(start)
  startw  = OMP_get_wtime()
  
  !$OMP PARALLEL DO  PRIVATE(j,x,y,z0)
  do i=1,Nx
     do j=1,Ny
        x = ext(1) + (ext(2)-ext(1))*(i-1.)/(Nx-1.)
        y = ext(3) + (ext(4)-ext(3))*(j-1.)/(Ny-1.)
        z0 = dcmplx(x,y)
        mande(i,j) = Mandelb(z0, max_steps)
     enddo
  enddo
  !$OMP END PARALLEL DO

  finishw = OMP_get_wtime()
  call cpu_time(finish)
  WRITE(0, '("clock time : ",f6.3,"s  wall time=",f6.3,"s")') finish-start, finishw-startw

  do i=1,Nx
     do j=1,Ny
        x = ext(1) + (ext(2)-ext(1))*(i-1.)/(Nx-1.)
        y = ext(3) + (ext(4)-ext(3))*(j-1.)/(Ny-1.)
        print *, x, y, 1./mande(i,j)
     enddo
  enddo
end program mand
