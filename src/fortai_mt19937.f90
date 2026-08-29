module fortai_mt19937
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private

    type, public :: mt19937_t
        private
        integer(int64) :: state(624) = 0_int64
        integer :: position = 625
    end type mt19937_t

    public :: mt19937_seed
    public :: mt19937_uint32
    public :: mt19937_uniform_real64

contains

    subroutine mt19937_seed(generator, seed)
        type(mt19937_t), intent(out) :: generator
        integer(int64), intent(in) :: seed
        integer(int64), parameter :: mask32 = int(z'ffffffff', int64)
        integer :: i

        generator%state(1) = iand(seed, mask32)
        do i = 2, size(generator%state)
            generator%state(i) = iand(1812433253_int64 * ieor(generator%state(i - 1), &
                shiftr(generator%state(i - 1), 30)) + int(i - 1, int64), mask32)
        end do
        generator%position = size(generator%state) + 1
    end subroutine mt19937_seed

    integer(int64) function mt19937_uint32(generator)
        type(mt19937_t), intent(inout) :: generator
        integer(int64), parameter :: mask32 = int(z'ffffffff', int64)
        integer(int64), parameter :: upper = int(z'80000000', int64)
        integer(int64), parameter :: lower = int(z'7fffffff', int64)
        integer(int64), parameter :: matrix = int(z'9908b0df', int64)
        integer(int64) :: value
        integer :: i, next, offset

        if (generator%position > size(generator%state)) then
            do i = 1, size(generator%state)
                next = i + 1
                if (next > size(generator%state)) next = 1
                offset = i + 397
                if (offset > size(generator%state)) offset = offset - size(generator%state)
                value = ior(iand(generator%state(i), upper), iand(generator%state(next), lower))
                generator%state(i) = ieor(generator%state(offset), shiftr(value, 1))
                if (btest(value, 0)) generator%state(i) = ieor(generator%state(i), matrix)
                generator%state(i) = iand(generator%state(i), mask32)
            end do
            generator%position = 1
        end if
        value = generator%state(generator%position)
        generator%position = generator%position + 1
        value = ieor(value, shiftr(value, 11))
        value = ieor(value, iand(shiftl(value, 7), int(z'9d2c5680', int64)))
        value = ieor(value, iand(shiftl(value, 15), int(z'efc60000', int64)))
        value = ieor(value, shiftr(value, 18))
        mt19937_uint32 = iand(value, mask32)
    end function mt19937_uint32

    real(real64) function mt19937_uniform_real64(generator)
        type(mt19937_t), intent(inout) :: generator
        real(real64), parameter :: radix32 = 4294967296.0_real64
        real(real64), parameter :: radix64 = 18446744073709551616.0_real64
        integer(int64) :: low, high

        ! libstdc++'s uniform_real_distribution<double> uses
        ! generate_canonical<double,53>, consuming two mt19937 words in this
        ! low-then-high order.
        low = mt19937_uint32(generator)
        high = mt19937_uint32(generator)
        mt19937_uniform_real64 = (real(low, real64) + real(high, real64) * radix32) / radix64
    end function mt19937_uniform_real64

end module fortai_mt19937
