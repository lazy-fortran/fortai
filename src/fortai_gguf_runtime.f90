module fortai_gguf_runtime
    use, intrinsic :: iso_c_binding, only: c_float, c_int8_t, c_int64_t
    use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, real64
    use fortai_status, only: FORTAI_INVALID, FORTAI_IO_ERROR, FORTAI_UNSUPPORTED, status_t
    implicit none
    private

    interface
        function fortai_q8_dot(weights, quantized, scales, row, block_count) &
                bind(C, name='fortai_q8_dot') result(value)
            import c_float, c_int8_t, c_int64_t
            integer(c_int8_t), intent(in) :: weights(*)
            integer(c_int8_t), intent(in) :: quantized(*)
            real(c_float), intent(in) :: scales(*)
            integer(c_int64_t), value, intent(in) :: row, block_count
            real(c_float) :: value
        end function fortai_q8_dot

        subroutine fortai_q8_quantize(vector, quantized, scales, count) &
                bind(C, name='fortai_q8_quantize')
            import c_float, c_int8_t, c_int64_t
            real(c_float), intent(in) :: vector(*)
            integer(c_int8_t), intent(out) :: quantized(*)
            real(c_float), intent(out) :: scales(*)
            integer(c_int64_t), value, intent(in) :: count
        end subroutine fortai_q8_quantize

        subroutine fortai_q8_dequantize_row(weights, values, block_count) &
                bind(C, name='fortai_q8_dequantize_row')
            import c_float, c_int8_t, c_int64_t
            integer(c_int8_t), intent(in) :: weights(*)
            real(c_float), intent(out) :: values(*)
            integer(c_int64_t), value, intent(in) :: block_count
        end subroutine fortai_q8_dequantize_row

    end interface

    integer(int32), parameter, public :: GGML_TYPE_F32 = 0_int32
    integer(int32), parameter, public :: GGML_TYPE_F16 = 1_int32
    integer(int32), parameter, public :: GGML_TYPE_Q8_0 = 8_int32

    integer(int32), parameter :: GGUF_UINT8 = 0_int32
    integer(int32), parameter :: GGUF_INT8 = 1_int32
    integer(int32), parameter :: GGUF_UINT16 = 2_int32
    integer(int32), parameter :: GGUF_INT16 = 3_int32
    integer(int32), parameter :: GGUF_UINT32 = 4_int32
    integer(int32), parameter :: GGUF_INT32 = 5_int32
    integer(int32), parameter :: GGUF_FLOAT32 = 6_int32
    integer(int32), parameter :: GGUF_BOOL = 7_int32
    integer(int32), parameter :: GGUF_STRING = 8_int32
    integer(int32), parameter :: GGUF_ARRAY = 9_int32
    integer(int32), parameter :: GGUF_UINT64 = 10_int32
    integer(int32), parameter :: GGUF_INT64 = 11_int32
    integer(int32), parameter :: GGUF_FLOAT64 = 12_int32

    type, public :: gguf_meta_t
        character(len=:), allocatable :: key
        integer(int32) :: value_type = -1_int32
        integer(int64) :: integer_value = 0_int64
        real(real32) :: real_value = 0.0_real32
        logical :: logical_value = .false.
        character(len=:), allocatable :: string_value
        integer(int32) :: element_type = -1_int32
        integer(int64), allocatable :: integer_values(:)
    end type gguf_meta_t

    type, public :: gguf_tensor_t
        character(len=:), allocatable :: name
        integer(int64), allocatable :: shape(:)
        integer(int32) :: value_type = -1_int32
        integer(int64) :: file_offset = 0_int64
        integer(int64) :: byte_count = 0_int64
        integer(int8), allocatable :: bytes(:)
        real(real32), allocatable :: decoded_values(:)
    contains
        procedure :: dot => gguf_tensor_dot
        procedure :: get_row => gguf_tensor_get_row
        procedure :: matvec => gguf_tensor_matvec
        procedure :: matvec_pair_q8 => gguf_tensor_matvec_pair_q8
        procedure :: matvec_triplet_q8 => gguf_tensor_matvec_triplet_q8
        procedure :: matvec_q8 => gguf_tensor_matvec_q8
        procedure :: value => gguf_tensor_value
    end type gguf_tensor_t

    type, public :: gguf_file_t
        integer(int32) :: version = 0_int32
        integer(int64) :: tensor_count = 0_int64
        integer(int64) :: metadata_count = 0_int64
        integer(int64) :: data_start = 0_int64
        integer(int64) :: alignment = 32_int64
        type(gguf_meta_t), allocatable :: metadata(:)
        type(gguf_tensor_t), allocatable :: tensors(:)
    contains
        procedure :: close => gguf_file_close
        procedure :: find_meta => gguf_file_find_meta
        procedure :: find_tensor => gguf_file_find_tensor
        procedure :: meta_int => gguf_file_meta_int
        procedure :: meta_real => gguf_file_meta_real
        procedure :: open => gguf_file_open
    end type gguf_file_t

    public :: gguf_fp16_to_real

contains

    subroutine gguf_file_open(self, path, stat)
        class(gguf_file_t), intent(inout) :: self
        character(len=*), intent(in) :: path
        type(status_t), intent(out) :: stat
        character(len=4) :: magic
        integer(int32) :: i, j, io_status, unit, value_type, rank32
        integer(int64) :: count, elements, position

        call self%close()
        call stat%clear()
        open (newunit=unit, file=path, access='stream', form='unformatted', &
            status='old', action='read', iostat=io_status)
        if (io_status /= 0) then
            call stat%set(FORTAI_IO_ERROR, 'could not open GGUF model')
            return
        end if

        read (unit, iostat=io_status) magic
        read (unit, iostat=io_status) self%version
        read (unit, iostat=io_status) self%tensor_count
        read (unit, iostat=io_status) self%metadata_count
        if (io_status /= 0 .or. magic /= 'GGUF' .or. self%version /= 3_int32) then
            close (unit)
            call stat%set(FORTAI_INVALID, 'unsupported or malformed GGUF header')
            return
        end if
        if (self%tensor_count < 0_int64 .or. self%metadata_count < 0_int64) then
            close (unit)
            call stat%set(FORTAI_INVALID, 'GGUF counts cannot be negative')
            return
        end if

        allocate (self%metadata(self%metadata_count))
        do i = 1, int(self%metadata_count)
            call read_string(unit, self%metadata(i)%key, io_status)
            read (unit, iostat=io_status) value_type
            self%metadata(i)%value_type = value_type
            call read_metadata_value(unit, value_type, self%metadata(i), io_status)
            if (io_status /= 0) then
                close (unit)
                call stat%set(FORTAI_IO_ERROR, 'could not read GGUF metadata')
                return
            end if
        end do

        self%alignment = self%meta_int('general.alignment', 32_int64)
        if (self%alignment <= 0_int64) self%alignment = 32_int64
        allocate (self%tensors(self%tensor_count))
        do i = 1, int(self%tensor_count)
            call read_string(unit, self%tensors(i)%name, io_status)
            read (unit, iostat=io_status) rank32
            count = int(rank32, int64)
            if (io_status /= 0 .or. count <= 0_int64 .or. count > 4_int64) then
                close (unit)
                call stat%set(FORTAI_INVALID, 'invalid GGUF tensor rank')
                return
            end if
            allocate (self%tensors(i)%shape(count))
            do j = 1, int(count)
                read (unit, iostat=io_status) self%tensors(i)%shape(j)
            end do
            read (unit, iostat=io_status) self%tensors(i)%value_type
            read (unit, iostat=io_status) self%tensors(i)%file_offset
            if (io_status /= 0) then
                close (unit)
                call stat%set(FORTAI_IO_ERROR, 'could not read GGUF tensor index')
                return
            end if
            elements = tensor_elements(self%tensors(i)%shape)
            self%tensors(i)%byte_count = tensor_bytes(self%tensors(i)%value_type, &
                self%tensors(i)%shape, elements, stat)
            if (.not. stat%is_ok()) then
                close (unit)
                return
            end if
        end do

        inquire (unit=unit, pos=position)
        self%data_start = align_up(position - 1_int64, self%alignment)
        do i = 1, int(self%tensor_count)
            allocate (self%tensors(i)%bytes(self%tensors(i)%byte_count))
            read (unit, pos=self%data_start + self%tensors(i)%file_offset + 1_int64, &
                iostat=io_status) self%tensors(i)%bytes
            if (io_status /= 0) then
                close (unit)
                call stat%set(FORTAI_IO_ERROR, 'could not read GGUF tensor data')
                return
            end if
            call decode_small_tensor(self%tensors(i))
        end do
        close (unit)
    end subroutine gguf_file_open

    subroutine gguf_file_close(self)
        class(gguf_file_t), intent(inout) :: self
        integer :: i

        if (allocated(self%tensors)) then
            do i = 1, size(self%tensors)
                if (allocated(self%tensors(i)%bytes)) deallocate (self%tensors(i)%bytes)
                if (allocated(self%tensors(i)%decoded_values)) &
                    deallocate (self%tensors(i)%decoded_values)
                if (allocated(self%tensors(i)%shape)) deallocate (self%tensors(i)%shape)
                if (allocated(self%tensors(i)%name)) deallocate (self%tensors(i)%name)
            end do
            deallocate (self%tensors)
        end if
        if (allocated(self%metadata)) then
            do i = 1, size(self%metadata)
                if (allocated(self%metadata(i)%integer_values)) &
                    deallocate (self%metadata(i)%integer_values)
                if (allocated(self%metadata(i)%string_value)) &
                    deallocate (self%metadata(i)%string_value)
                if (allocated(self%metadata(i)%key)) deallocate (self%metadata(i)%key)
            end do
            deallocate (self%metadata)
        end if
        self%version = 0_int32
        self%tensor_count = 0_int64
        self%metadata_count = 0_int64
        self%data_start = 0_int64
    end subroutine gguf_file_close

    integer function gguf_file_find_meta(self, key)
        class(gguf_file_t), intent(in) :: self
        character(len=*), intent(in) :: key
        integer :: i

        gguf_file_find_meta = 0
        if (.not. allocated(self%metadata)) return
        do i = 1, size(self%metadata)
            if (self%metadata(i)%key == key) then
                gguf_file_find_meta = i
                return
            end if
        end do
    end function gguf_file_find_meta

    integer function gguf_file_find_tensor(self, name)
        class(gguf_file_t), intent(in) :: self
        character(len=*), intent(in) :: name
        integer :: i

        gguf_file_find_tensor = 0
        if (.not. allocated(self%tensors)) return
        do i = 1, size(self%tensors)
            if (self%tensors(i)%name == name) then
                gguf_file_find_tensor = i
                return
            end if
        end do
    end function gguf_file_find_tensor

    integer(int64) function gguf_file_meta_int(self, key, default)
        class(gguf_file_t), intent(in) :: self
        character(len=*), intent(in) :: key
        integer(int64), intent(in) :: default
        integer :: index

        gguf_file_meta_int = default
        index = self%find_meta(key)
        if (index == 0) return
        select case (self%metadata(index)%value_type)
        case (GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16, GGUF_UINT32, &
                GGUF_INT32, GGUF_UINT64, GGUF_INT64)
            gguf_file_meta_int = self%metadata(index)%integer_value
        end select
    end function gguf_file_meta_int

    real(real32) function gguf_file_meta_real(self, key, default)
        class(gguf_file_t), intent(in) :: self
        character(len=*), intent(in) :: key
        real(real32), intent(in) :: default
        integer :: index

        gguf_file_meta_real = default
        index = self%find_meta(key)
        if (index == 0) return
        select case (self%metadata(index)%value_type)
        case (GGUF_FLOAT32, GGUF_FLOAT64)
            gguf_file_meta_real = self%metadata(index)%real_value
        end select
    end function gguf_file_meta_real

    real(real32) function gguf_tensor_value(self, index)
        class(gguf_tensor_t), intent(in) :: self
        integer(int64), intent(in) :: index
        integer(int64) :: block, byte_index, within
        integer(int16) :: scale_bits

        gguf_tensor_value = 0.0_real32
        if (.not. allocated(self%bytes)) return
        if (index < 1_int64 .or. index > tensor_elements(self%shape)) return
        if (allocated(self%decoded_values)) then
            gguf_tensor_value = self%decoded_values(index)
            return
        end if
        select case (self%value_type)
        case (GGML_TYPE_F32)
            byte_index = 1_int64 + (index - 1_int64) * 4_int64
            gguf_tensor_value = transfer(self%bytes(byte_index:byte_index + 3_int64), &
                gguf_tensor_value)
        case (GGML_TYPE_F16)
            byte_index = 1_int64 + (index - 1_int64) * 2_int64
            gguf_tensor_value = gguf_fp16_to_real(transfer( &
                self%bytes(byte_index:byte_index + 1_int64), 0_int16))
        case (GGML_TYPE_Q8_0)
            block = (index - 1_int64) / 32_int64
                within = mod(index - 1_int64, 32_int64)
                byte_index = block * 34_int64 + 1_int64
                scale_bits = transfer(self%bytes(byte_index:byte_index + 1_int64), scale_bits)
                gguf_tensor_value = gguf_fp16_to_real(scale_bits) * real( &
                    int(self%bytes(byte_index + 2_int64 + within), int32), real32)
            case default
                gguf_tensor_value = 0.0_real32
            end select
        end function gguf_tensor_value

        subroutine gguf_tensor_get_row(self, row, values, stat)
            class(gguf_tensor_t), intent(in) :: self
            integer(int64), intent(in) :: row
            real(real32), contiguous, intent(out) :: values(:)
            type(status_t), intent(out) :: stat
            integer(int64) :: i, width

            call stat%clear()
            if (size(self%shape) /= 2) then
                call stat%set(FORTAI_INVALID, 'GGUF tensor is not a matrix')
                return
            end if
            width = self%shape(1)
            if (row < 1_int64 .or. row > self%shape(2) .or. size(values) /= width) then
                call stat%set(FORTAI_INVALID, 'GGUF tensor row has the wrong shape')
                return
            end if
            if (self%value_type == GGML_TYPE_Q8_0) then
                call fortai_q8_dequantize_row(self%bytes((row - 1_int64) * &
                    (width / 32_int64) * 34_int64 + 1_int64:), values, width / 32_int64)
                return
            end if
            do i = 1, width
                values(i) = self%value((row - 1_int64) * width + i)
            end do
        end subroutine gguf_tensor_get_row

        real(real32) function gguf_tensor_dot(self, row, vector)
            class(gguf_tensor_t), intent(in) :: self
            integer(int64), intent(in) :: row
            real(real32), intent(in) :: vector(:)
            integer(int64) :: block, block_count, i, offset, width
            real(real32) :: accumulator, scale
            integer(int16) :: scale_bits
            integer(int32) :: quantized

            gguf_tensor_dot = 0.0_real32
            if (size(self%shape) /= 2) return
            width = self%shape(1)
            if (row < 1_int64 .or. row > self%shape(2) .or. size(vector) /= width) return
            select case (self%value_type)
            case (GGML_TYPE_Q8_0)
                block_count = width / 32_int64
                offset = (row - 1_int64) * block_count * 34_int64 + 1_int64
                accumulator = 0.0_real32
                do block = 0, block_count - 1_int64
                    scale_bits = transfer(self%bytes(offset + block * 34_int64: &
                        offset + block * 34_int64 + 1_int64), scale_bits)
                    scale = gguf_fp16_to_real(scale_bits)
                    !$omp simd reduction(+:accumulator)
                    do i = 0, 31
                        quantized = int(self%bytes(offset + block * 34_int64 + 2_int64 + i), &
                            int32)
                        accumulator = accumulator + scale * real(quantized, real32) * &
                            vector(block * 32_int64 + i + 1_int64)
                    end do
                end do
                gguf_tensor_dot = accumulator
            case (GGML_TYPE_F32, GGML_TYPE_F16)
                do i = 1, width
                    gguf_tensor_dot = gguf_tensor_dot + self%value((row - 1_int64) * width + i) &
                        * vector(i)
                end do
            end select
        end function gguf_tensor_dot

        subroutine gguf_tensor_matvec(self, vector, values, stat)
            class(gguf_tensor_t), intent(in) :: self
            real(real32), contiguous, intent(in) :: vector(:)
            real(real32), contiguous, intent(out) :: values(:)
            type(status_t), intent(out) :: stat
            integer(int64) :: row

            call stat%clear()
            if (size(self%shape) /= 2) then
                call stat%set(FORTAI_INVALID, 'GGUF matvec dimensions do not agree')
                return
            end if
            if (size(vector) /= self%shape(1) .or. size(values) /= self%shape(2)) then
                call stat%set(FORTAI_INVALID, 'GGUF matvec dimensions do not agree')
                return
            end if
            if (self%value_type /= GGML_TYPE_Q8_0 .and. self%value_type /= GGML_TYPE_F32 &
                .and. self%value_type /= GGML_TYPE_F16) then
                call stat%set(FORTAI_UNSUPPORTED, 'GGUF matvec type is not supported')
                return
            end if
            if (self%shape(2) < 128_int64) then
                do row = 1, self%shape(2)
                    values(row) = self%dot(row, vector)
                end do
            else
                !$omp parallel do default(none) shared(self, vector, values) private(row) schedule(static)
                do row = 1, self%shape(2)
                    values(row) = self%dot(row, vector)
                end do
                !$omp end parallel do
            end if
        end subroutine gguf_tensor_matvec

        subroutine gguf_tensor_matvec_q8(self, vector, values, quantized, scales, stat, task_parallel)
            class(gguf_tensor_t), intent(in) :: self
            real(real32), contiguous, intent(in) :: vector(:)
            real(real32), contiguous, intent(out) :: values(:)
            integer(int8), contiguous, intent(out) :: quantized(:)
            real(real32), contiguous, intent(out) :: scales(:)
            type(status_t), intent(out) :: stat
            logical, intent(in), optional :: task_parallel
            integer(int64) :: row, block_count

            call stat%clear()
            if (self%value_type /= GGML_TYPE_Q8_0) then
                call stat%set(FORTAI_UNSUPPORTED, 'activation Q8 matvec needs a Q8_0 tensor')
                return
            end if
            if (size(self%shape) /= 2) then
                call stat%set(FORTAI_INVALID, 'activation Q8 matvec dimensions do not agree')
                return
            end if
            if (size(vector) /= self%shape(1) .or. size(values) /= self%shape(2) .or. &
                mod(size(vector), 32) /= 0 .or. &
                size(quantized) < size(vector) + 2 * ((size(vector) + 31) / 32) .or. &
                size(scales) < (size(vector) + 31) / 32) then
                call stat%set(FORTAI_INVALID, 'activation Q8 matvec dimensions do not agree')
                return
            end if
            block_count = size(vector) / 32
            call fortai_q8_quantize(vector, quantized, scales, int(size(vector), c_int64_t))
            if (self%shape(2) < 128_int64) then
                do row = 1, self%shape(2)
                    values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                        int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                end do
            else if (present(task_parallel)) then
                if (task_parallel) then
                    !$omp taskloop default(none) shared(self, quantized, scales, values, block_count) &
                    !$omp& private(row) grainsize(256)
                    do row = 1, self%shape(2)
                        values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    end do
                    !$omp end taskloop
                else
                    !$omp parallel do default(none) shared(self, quantized, scales, values, block_count) &
                    !$omp& private(row) schedule(static)
                    do row = 1, self%shape(2)
                        values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    end do
                    !$omp end parallel do
                end if
            else
                !$omp parallel do default(none) shared(self, quantized, scales, values, block_count) &
                !$omp& private(row) schedule(static)
                do row = 1, self%shape(2)
                    values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                        int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                end do
                !$omp end parallel do
            end if
        end subroutine gguf_tensor_matvec_q8

        subroutine gguf_tensor_matvec_pair_q8(self, other, vector, values, other_values, &
                quantized, scales, stat, task_parallel)
            class(gguf_tensor_t), intent(in) :: self, other
            real(real32), contiguous, intent(in) :: vector(:)
            real(real32), contiguous, intent(out) :: values(:), other_values(:)
            integer(int8), contiguous, intent(out) :: quantized(:)
            real(real32), contiguous, intent(out) :: scales(:)
            type(status_t), intent(out) :: stat
            logical, intent(in), optional :: task_parallel
            integer(int64) :: row, first_rows, total_rows, block_count

            call stat%clear()
            if (self%value_type /= GGML_TYPE_Q8_0 .or. other%value_type /= GGML_TYPE_Q8_0) then
                call stat%set(FORTAI_UNSUPPORTED, 'paired activation matvec needs Q8_0 tensors')
                return
            end if
            if (size(self%shape) /= 2 .or. size(other%shape) /= 2) then
                call stat%set(FORTAI_INVALID, 'paired activation matvec dimensions do not agree')
                return
            end if
            if (self%shape(1) /= other%shape(1) .or. &
                size(vector) /= self%shape(1) .or. size(values) /= self%shape(2) .or. &
                size(other_values) /= other%shape(2) .or. mod(size(vector), 32) /= 0 .or. &
                size(quantized) < size(vector) + 2 * ((size(vector) + 31) / 32) .or. &
                size(scales) < (size(vector) + 31) / 32) then
                call stat%set(FORTAI_INVALID, 'paired activation matvec dimensions do not agree')
                return
            end if
            block_count = size(vector) / 32
            call fortai_q8_quantize(vector, quantized, scales, int(size(vector), c_int64_t))
            first_rows = self%shape(2)
            total_rows = first_rows + other%shape(2)
            if (total_rows < 128_int64) then
                do row = 1, total_rows
                    if (row <= first_rows) then
                        values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    else
                        other_values(row - first_rows) = fortai_q8_dot(other%bytes, quantized, scales, &
                            int(row - first_rows - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    end if
                end do
            else if (present(task_parallel)) then
                if (task_parallel) then
                    !$omp taskloop default(none) shared(self, other, quantized, scales, values, &
                    !$omp& other_values, first_rows, total_rows, block_count) private(row) grainsize(256)
                    do row = 1, total_rows
                        if (row <= first_rows) then
                            values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                                int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        else
                            other_values(row - first_rows) = fortai_q8_dot(other%bytes, quantized, scales, &
                                int(row - first_rows - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        end if
                    end do
                    !$omp end taskloop
                else
                    !$omp parallel do default(none) shared(self, other, quantized, scales, values, &
                    !$omp& other_values, first_rows, total_rows, block_count) private(row) schedule(static)
                    do row = 1, total_rows
                        if (row <= first_rows) then
                            values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                                int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        else
                            other_values(row - first_rows) = fortai_q8_dot(other%bytes, quantized, scales, &
                                int(row - first_rows - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        end if
                    end do
                    !$omp end parallel do
                end if
            else
                !$omp parallel do default(none) shared(self, other, quantized, scales, values, &
                !$omp& other_values, first_rows, total_rows, block_count) private(row) schedule(static)
                do row = 1, total_rows
                    if (row <= first_rows) then
                        values(row) = fortai_q8_dot(self%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    else
                        other_values(row - first_rows) = fortai_q8_dot(other%bytes, quantized, scales, &
                            int(row - first_rows - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    end if
                end do
                !$omp end parallel do
            end if
        end subroutine gguf_tensor_matvec_pair_q8

        subroutine gguf_tensor_matvec_triplet_q8(self, second, third, vector, values, &
                second_values, third_values, quantized, scales, stat, task_parallel)
            class(gguf_tensor_t), intent(in) :: self, second, third
            real(real32), contiguous, intent(in) :: vector(:)
            real(real32), contiguous, intent(out) :: values(:), second_values(:), third_values(:)
            integer(int8), contiguous, intent(out) :: quantized(:)
            real(real32), contiguous, intent(out) :: scales(:)
            type(status_t), intent(out) :: stat
            logical, intent(in), optional :: task_parallel
            integer(int64) :: row, total_rows, first_rows, second_rows
            integer(int64) :: block_count, index

            call stat%clear()
            if (self%value_type /= GGML_TYPE_Q8_0 .or. second%value_type /= GGML_TYPE_Q8_0 &
                .or. third%value_type /= GGML_TYPE_Q8_0) then
                call stat%set(FORTAI_UNSUPPORTED, 'triplet activation matvec needs Q8_0 tensors')
                return
            end if
            if (size(self%shape) /= 2 .or. size(second%shape) /= 2 .or. &
                size(third%shape) /= 2) then
                call stat%set(FORTAI_INVALID, 'triplet matvec dimensions do not agree')
                return
            end if
            if (self%shape(1) /= second%shape(1) .or. self%shape(1) /= third%shape(1) .or. &
                size(vector) /= self%shape(1) .or. size(values) /= self%shape(2) .or. &
                size(second_values) /= second%shape(2) .or. &
                size(third_values) /= third%shape(2) .or. mod(size(vector), 32) /= 0 .or. &
                size(quantized) < size(vector) + 2 * ((size(vector) + 31) / 32) .or. &
                size(scales) < (size(vector) + 31) / 32) then
                call stat%set(FORTAI_INVALID, 'triplet matvec dimensions do not agree')
                return
            end if
            block_count = size(vector) / 32
            call fortai_q8_quantize(vector, quantized, scales, int(size(vector), c_int64_t))

            first_rows = self%shape(2)
            second_rows = second%shape(2)
            total_rows = first_rows + second_rows + third%shape(2)
            if (total_rows < 128_int64) then
                do index = 1, total_rows
                    if (index <= first_rows) then
                        values(index) = fortai_q8_dot(self%bytes, quantized, scales, &
                            int(index - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    else if (index <= first_rows + second_rows) then
                        row = index - first_rows
                        second_values(row) = fortai_q8_dot(second%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    else
                        row = index - first_rows - second_rows
                        third_values(row) = fortai_q8_dot(third%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    end if
                end do
            else if (present(task_parallel)) then
                if (task_parallel) then
                    !$omp taskloop default(none) shared(self, second, third, quantized, scales, &
                    !$omp& values, second_values, third_values, first_rows, second_rows, total_rows, &
                    !$omp& block_count) private(index, row) grainsize(256)
                    do index = 1, total_rows
                        if (index <= first_rows) then
                            values(index) = fortai_q8_dot(self%bytes, quantized, scales, &
                                int(index - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        else if (index <= first_rows + second_rows) then
                            row = index - first_rows
                            second_values(row) = fortai_q8_dot(second%bytes, quantized, scales, &
                                int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        else
                            row = index - first_rows - second_rows
                            third_values(row) = fortai_q8_dot(third%bytes, quantized, scales, &
                                int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        end if
                    end do
                    !$omp end taskloop
                else
                    !$omp parallel do default(none) shared(self, second, third, quantized, scales, &
                    !$omp& values, second_values, third_values, first_rows, second_rows, total_rows, &
                    !$omp& block_count) private(index, row) schedule(static)
                    do index = 1, total_rows
                        if (index <= first_rows) then
                            values(index) = fortai_q8_dot(self%bytes, quantized, scales, &
                                int(index - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        else if (index <= first_rows + second_rows) then
                            row = index - first_rows
                            second_values(row) = fortai_q8_dot(second%bytes, quantized, scales, &
                                int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        else
                            row = index - first_rows - second_rows
                            third_values(row) = fortai_q8_dot(third%bytes, quantized, scales, &
                                int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                        end if
                    end do
                    !$omp end parallel do
                end if
            else
                !$omp parallel do default(none) shared(self, second, third, quantized, scales, &
                !$omp& values, second_values, third_values, first_rows, second_rows, total_rows, &
                !$omp& block_count) private(index, row) schedule(static)
                do index = 1, total_rows
                    if (index <= first_rows) then
                        values(index) = fortai_q8_dot(self%bytes, quantized, scales, &
                            int(index - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    else if (index <= first_rows + second_rows) then
                        row = index - first_rows
                        second_values(row) = fortai_q8_dot(second%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    else
                        row = index - first_rows - second_rows
                        third_values(row) = fortai_q8_dot(third%bytes, quantized, scales, &
                            int(row - 1_int64, c_int64_t), int(block_count, c_int64_t))
                    end if
                end do
                !$omp end parallel do
            end if
        end subroutine gguf_tensor_matvec_triplet_q8

        subroutine read_metadata_value(unit, value_type, value, io_status)
            integer, intent(in) :: unit
            integer(int32), intent(in) :: value_type
            type(gguf_meta_t), intent(inout) :: value
            integer, intent(out) :: io_status
            integer(int8) :: byte_value
            integer(int16) :: short_value
            integer(int32) :: element_type, i32
            integer(int64) :: count, i
            real(real64) :: r64

            io_status = 0
            select case (value_type)
            case (GGUF_UINT8)
                read (unit, iostat=io_status) byte_value
                value%integer_value = iand(int(byte_value, int64), 255_int64)
            case (GGUF_INT8)
                read (unit, iostat=io_status) byte_value
                value%integer_value = int(byte_value, int64)
            case (GGUF_UINT16)
                read (unit, iostat=io_status) short_value
                value%integer_value = iand(int(short_value, int64), 65535_int64)
            case (GGUF_INT16)
                read (unit, iostat=io_status) short_value
                value%integer_value = int(short_value, int64)
            case (GGUF_INT32)
                read (unit, iostat=io_status) i32
                value%integer_value = i32
            case (GGUF_UINT32)
                read (unit, iostat=io_status) i32
                value%integer_value = iand(int(i32, int64), int(z'ffffffff', int64))
            case (GGUF_UINT64, GGUF_INT64)
                read (unit, iostat=io_status) value%integer_value
            case (GGUF_FLOAT32)
                read (unit, iostat=io_status) value%real_value
            case (GGUF_FLOAT64)
                read (unit, iostat=io_status) r64
                value%real_value = real(r64, real32)
            case (GGUF_BOOL)
                read (unit, iostat=io_status) byte_value
                value%logical_value = byte_value /= 0_int8
            case (GGUF_STRING)
                call read_string(unit, value%string_value, io_status)
            case (GGUF_ARRAY)
                read (unit, iostat=io_status) element_type
                read (unit, iostat=io_status) count
                value%element_type = element_type
                if (io_status /= 0) return
                if (element_type == GGUF_INT32 .or. element_type == GGUF_UINT32 .or. &
                    element_type == GGUF_INT64 .or. element_type == GGUF_UINT64) then
                    allocate (value%integer_values(count))
                    do i = 1, count
                        if (element_type == GGUF_INT32 .or. element_type == GGUF_UINT32) then
                            read (unit, iostat=io_status) i32
                            value%integer_values(i) = i32
                        else
                            read (unit, iostat=io_status) value%integer_values(i)
                        end if
                    end do
                else
                    do i = 1, count
                        call skip_metadata_value(unit, element_type, io_status)
                        if (io_status /= 0) return
                    end do
                end if
            case default
                io_status = 1
            end select
        end subroutine read_metadata_value

        subroutine skip_metadata_value(unit, value_type, io_status)
            integer, intent(in) :: unit
            integer(int32), intent(in) :: value_type
            integer, intent(out) :: io_status
            character(len=:), allocatable :: ignored
            integer(int8) :: byte_value
            integer(int16) :: short_value
            integer(int32) :: i32
            integer(int64) :: i64
            real(real32) :: r32
            real(real64) :: r64
            logical :: flag

            io_status = 0
            select case (value_type)
            case (GGUF_UINT8, GGUF_INT8, GGUF_BOOL)
                read (unit, iostat=io_status) byte_value
            case (GGUF_UINT16, GGUF_INT16)
                read (unit, iostat=io_status) short_value
            case (GGUF_UINT32, GGUF_INT32)
                read (unit, iostat=io_status) i32
            case (GGUF_FLOAT32)
                read (unit, iostat=io_status) r32
            case (GGUF_STRING)
                call read_string(unit, ignored, io_status)
            case (GGUF_UINT64, GGUF_INT64)
                read (unit, iostat=io_status) i64
            case (GGUF_FLOAT64)
                read (unit, iostat=io_status) r64
            case default
                io_status = 1
            end select
        end subroutine skip_metadata_value

        subroutine read_string(unit, value, io_status)
            integer, intent(in) :: unit
            character(len=:), allocatable, intent(out) :: value
            integer, intent(out) :: io_status
            integer(int64) :: length

            read (unit, iostat=io_status) length
            if (io_status /= 0 .or. length < 0_int64) return
            allocate (character(len=length) :: value)
            if (length > 0_int64) read (unit, iostat=io_status) value
        end subroutine read_string

        subroutine decode_small_tensor(tensor)
            type(gguf_tensor_t), intent(inout) :: tensor
            integer(int64) :: byte_index, elements, index

            if (tensor%value_type /= GGML_TYPE_F32 .and. tensor%value_type /= GGML_TYPE_F16) return
            elements = tensor_elements(tensor%shape)
            if (elements > 131072_int64) return
            allocate (tensor%decoded_values(elements))
            if (tensor%value_type == GGML_TYPE_F32) then
                do index = 1, elements
                    byte_index = 1_int64 + (index - 1_int64) * 4_int64
                    tensor%decoded_values(index) = transfer( &
                        tensor%bytes(byte_index:byte_index + 3_int64), 0.0_real32)
                end do
            else
                do index = 1, elements
                    byte_index = 1_int64 + (index - 1_int64) * 2_int64
                    tensor%decoded_values(index) = gguf_fp16_to_real(transfer( &
                        tensor%bytes(byte_index:byte_index + 1_int64), 0_int16))
                end do
            end if
        end subroutine decode_small_tensor

        integer(int64) function tensor_elements(shape)
            integer(int64), intent(in) :: shape(:)
            integer :: i

            tensor_elements = 1_int64
            do i = 1, size(shape)
                tensor_elements = tensor_elements * shape(i)
            end do
        end function tensor_elements

        integer(int64) function tensor_bytes(value_type, shape, elements, stat)
            integer(int32), intent(in) :: value_type
            integer(int64), intent(in) :: shape(:)
            integer(int64), intent(in) :: elements
            type(status_t), intent(out) :: stat

            call stat%clear()
            select case (value_type)
            case (GGML_TYPE_F32)
                tensor_bytes = 4_int64 * elements
            case (GGML_TYPE_F16)
                tensor_bytes = 2_int64 * elements
            case (GGML_TYPE_Q8_0)
                if (shape(1) <= 0_int64 .or. mod(shape(1), 32_int64) /= 0_int64) then
                    call stat%set(FORTAI_INVALID, 'Q8_0 tensor width is not a multiple of 32')
                    tensor_bytes = 0_int64
                    return
                end if
                tensor_bytes = (elements / shape(1)) * (shape(1) / 32_int64) * 34_int64
            case default
                call stat%set(FORTAI_UNSUPPORTED, 'GGUF tensor quantization is not supported')
                tensor_bytes = 0_int64
            end select
        end function tensor_bytes

        integer(int64) function align_up(value, alignment)
            integer(int64), intent(in) :: value, alignment

            align_up = ((value + alignment - 1_int64) / alignment) * alignment
        end function align_up

        real(real32) function gguf_fp16_to_real(bits)
            integer(int16), intent(in) :: bits
            integer(int32) :: raw, exponent, fraction, sign

            raw = iand(int(bits, int32), int(z'ffff', int32))
            sign = iand(raw, int(z'8000', int32))
            exponent = iand(ishft(raw, -10), 31_int32)
            fraction = iand(raw, 1023_int32)
            if (exponent == 0) then
                gguf_fp16_to_real = real(fraction, real32) / 1024.0_real32
                gguf_fp16_to_real = gguf_fp16_to_real * 2.0_real32 ** (-14)
            else if (exponent == 31) then
                gguf_fp16_to_real = huge(1.0_real32)
            else
                gguf_fp16_to_real = (1.0_real32 + real(fraction, real32) / 1024.0_real32) * &
                    2.0_real32 ** (exponent - 15)
            end if
            if (sign /= 0) gguf_fp16_to_real = -gguf_fp16_to_real
        end function gguf_fp16_to_real

    end module fortai_gguf_runtime
