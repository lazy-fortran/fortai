module fortai_whisper_audio
    !! Whisper's exact log-mel frontend, owned by FortAI.
    !!
    !! The reference implementation uses a radix-2/DFT fallback FFT and a
    !! double accumulator for each mel band.  This implementation preserves
    !! those numerical choices first; a CUDA cuFFT/fused implementation can be
    !! added behind the same interface only after an oracle comparison.
    use, intrinsic :: iso_fortran_env, only: int32, real32, real64
    use fortai_status, only: FORTAI_INVALID, status_t
    use fortai_whisper_format, only: WHISPER_HOP_LENGTH, WHISPER_N_FFT, WHISPER_SAMPLE_RATE
    implicit none
    private

    type, public :: whisper_mel_t
        integer(int32) :: n_mel = 0_int32
        integer(int32) :: n_len = 0_int32
        integer(int32) :: n_len_org = 0_int32
        real(real32), allocatable :: data(:,:)
    contains
        procedure :: clear => whisper_mel_clear
    end type whisper_mel_t

    public :: whisper_log_mel_spectrogram
    public :: whisper_hann_window

    real(real32), parameter :: PI32 = 3.1415926535897932384626433832795_real32
    real(real64), parameter :: LOG10_EPS = 1.0e-10_real64

contains

    subroutine whisper_mel_clear(self)
        class(whisper_mel_t), intent(inout) :: self

        if (allocated(self%data)) deallocate(self%data)
        self%n_mel = 0_int32
        self%n_len = 0_int32
        self%n_len_org = 0_int32
    end subroutine whisper_mel_clear

    subroutine whisper_hann_window(window)
        real(real32), allocatable, intent(out) :: window(:)
        integer :: i

        allocate(window(WHISPER_N_FFT))
        do i = 0, WHISPER_N_FFT - 1
            ! The C++ reference uses cosf and a periodic window.
            window(i + 1) = 0.5_real32 * (1.0_real32 - &
                cos((2.0_real32 * PI32 * real(i, real32)) / real(WHISPER_N_FFT, real32)))
        end do
    end subroutine whisper_hann_window

    subroutine whisper_log_mel_spectrogram(samples, filters, mel, n_threads, stat)
        real(real32), intent(in) :: samples(:)
        real(real32), intent(in) :: filters(:,:)
        type(whisper_mel_t), intent(inout) :: mel
        integer(int32), intent(in), optional :: n_threads
        type(status_t), intent(out) :: stat
        real(real32), allocatable :: samples_padded(:), hann(:), sin_table(:), cos_table(:)
        integer(int32) :: threads, n_samples, n_samples_effective
        integer(int32) :: stage_1_pad, stage_2_pad, padded_size
        integer(int32) :: n_len, n_len_org
        integer :: i, j, k, source_index
        real(real64) :: mmax

        call stat%clear()
        call mel%clear()
        n_samples = int(size(samples), int32)
        if (n_samples < 0) then
            call stat%set(FORTAI_INVALID, 'Whisper PCM length is invalid')
            return
        end if
        if (size(filters, 1) <= 0 .or. size(filters, 2) /= 1 + WHISPER_N_FFT / 2) then
            call stat%set(FORTAI_INVALID, 'Whisper mel filter shape is invalid')
            return
        end if
        threads = 1_int32
        if (present(n_threads)) threads = max(1_int32, n_threads)

        stage_1_pad = WHISPER_SAMPLE_RATE * 30_int32
        stage_2_pad = WHISPER_N_FFT / 2_int32
        padded_size = n_samples + stage_1_pad + 2_int32 * stage_2_pad
        n_samples_effective = n_samples + stage_2_pad
        n_len = (padded_size - WHISPER_N_FFT) / WHISPER_HOP_LENGTH
        n_len_org = 1_int32 + (n_samples + stage_2_pad - WHISPER_N_FFT) / WHISPER_HOP_LENGTH
        if (n_len <= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper PCM is too short for a mel spectrogram')
            return
        end if

        allocate(samples_padded(padded_size))
        samples_padded = 0.0_real32
        if (n_samples > 0) samples_padded(stage_2_pad + 1:stage_2_pad + n_samples) = samples

        ! Reflect the first 200 samples exactly as reverse_copy(samples+1,...).
        do i = 1, stage_2_pad
            if (n_samples >= 2) then
                source_index = min(n_samples, i + 1)
            else if (n_samples == 1) then
                source_index = 1
            else
                source_index = 0
            end if
            if (source_index > 0) samples_padded(stage_2_pad - i + 1) = samples(source_index)
        end do

        call whisper_hann_window(hann)
        allocate(sin_table(WHISPER_N_FFT), cos_table(WHISPER_N_FFT))
        do i = 0, WHISPER_N_FFT - 1
            sin_table(i + 1) = sin((2.0_real32 * PI32 * real(i, real32)) / real(WHISPER_N_FFT, real32))
            cos_table(i + 1) = cos((2.0_real32 * PI32 * real(i, real32)) / real(WHISPER_N_FFT, real32))
        end do

        mel%n_mel = int(size(filters, 1), int32)
        mel%n_len = n_len
        mel%n_len_org = n_len_org
        allocate(mel%data(mel%n_mel, mel%n_len))
        mel%data = real(log10(LOG10_EPS), real32)

        call whisper_compute_frames(samples_padded, n_samples_effective, filters, hann, sin_table, cos_table, &
            threads, mel, stat)
        if (.not. stat%is_ok()) then
            call mel%clear()
            deallocate(samples_padded, hann, sin_table, cos_table)
            return
        end if

        mmax = maxval(real(mel%data, real64))
        mmax = mmax - 8.0_real64
        do k = 1, size(mel%data, 2)
            do j = 1, size(mel%data, 1)
                mel%data(j, k) = real((max(real(mel%data(j, k), real64), mmax) + 4.0_real64) / &
                    4.0_real64, real32)
            end do
        end do
        deallocate(samples_padded, hann, sin_table, cos_table)
    end subroutine whisper_log_mel_spectrogram

    subroutine whisper_compute_frames(samples, n_samples, filters, hann, sin_table, cos_table, threads, mel, stat)
        real(real32), intent(in) :: samples(:), filters(:,:), hann(:), sin_table(:), cos_table(:)
        integer(int32), intent(in) :: n_samples, threads
        type(whisper_mel_t), intent(inout) :: mel
        type(status_t), intent(out) :: stat
        integer(int32) :: frame_limit
        integer :: frame, index

        call stat%clear()
        if (threads <= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper audio worker count is invalid')
            return
        end if
        frame_limit = min(n_samples / WHISPER_HOP_LENGTH + 1_int32, mel%n_len)

        ! Each worker owns its FFT scratch.  The frame output is disjoint, so
        ! no locks or atomics are needed in the hot loop.
        !$omp parallel default(none) num_threads(threads) shared(samples, n_samples, filters, hann, sin_table, cos_table, &
        !$omp& threads, mel, frame_limit) private(frame)
        block
            real(real32), allocatable :: fft_in(:), fft_out(:)
            integer :: j, k, offset, valid
            real(real64) :: sum

            allocate(fft_in(2 * WHISPER_N_FFT), fft_out(8 * WHISPER_N_FFT))
            !$omp do schedule(static)
            do frame = 0, frame_limit - 1
                offset = frame * WHISPER_HOP_LENGTH
                fft_in = 0.0_real32
                valid = min(WHISPER_N_FFT, n_samples - offset)
                if (valid > 0) then
                    do j = 0, valid - 1
                        fft_in(j + 1) = hann(j + 1) * samples(offset + j + 1)
                    end do
                end if
                call whisper_fft(fft_in, 0, WHISPER_N_FFT, fft_out, 0, sin_table, cos_table)
                do j = 0, size(filters, 2) - 1
                    fft_out(j + 1) = fft_out(2 * j + 1) * fft_out(2 * j + 1) + &
                        fft_out(2 * j + 2) * fft_out(2 * j + 2)
                end do
                do j = 1, size(filters, 1)
                    sum = 0.0_real64
                    k = 0
                    do while (k + 3 < size(filters, 2))
                        sum = sum + real(fft_out(k + 1), real64) * real(filters(j, k + 1), real64) + &
                            real(fft_out(k + 2), real64) * real(filters(j, k + 2), real64) + &
                            real(fft_out(k + 3), real64) * real(filters(j, k + 3), real64) + &
                            real(fft_out(k + 4), real64) * real(filters(j, k + 4), real64)
                        k = k + 4
                    end do
                    do while (k < size(filters, 2))
                        sum = sum + real(fft_out(k + 1), real64) * real(filters(j, k + 1), real64)
                        k = k + 1
                    end do
                    mel%data(j, frame + 1) = real(log10(max(sum, LOG10_EPS)), real32)
                end do
            end do
            !$omp end do
            deallocate(fft_in, fft_out)
        end block
        !$omp end parallel

        ! Frames beyond the real input are already initialized to log10(eps).
        index = frame_limit + 1
        if (index <= mel%n_len) then
            mel%data(:, index:mel%n_len) = real(log10(LOG10_EPS), real32)
        end if
    end subroutine whisper_compute_frames

    recursive subroutine whisper_fft(input, input_base, n, output, output_base, sin_table, cos_table)
        real(real32), intent(inout) :: input(:)
        integer, intent(in) :: input_base, n
        real(real32), intent(inout) :: output(:)
        integer, intent(in) :: output_base
        real(real32), intent(in) :: sin_table(:), cos_table(:)
        integer :: half_n, i, k, even_base, even_fft_base, odd_fft_base, index, step
        real(real32) :: re, im, even_re, even_im, odd_re, odd_im

        if (n <= 1) then
            output(output_base + 1) = input(input_base + 1)
            output(output_base + 2) = 0.0_real32
            return
        end if
        half_n = n / 2
        if (n - 2 * half_n == 1) then
            step = WHISPER_N_FFT / n
            do k = 0, n - 1
                re = 0.0_real32
                im = 0.0_real32
                do i = 0, n - 1
                    index = mod(k * i * step, WHISPER_N_FFT)
                    re = re + input(input_base + i + 1) * cos_table(index + 1)
                    im = im - input(input_base + i + 1) * sin_table(index + 1)
                end do
                output(output_base + 2 * k + 1) = re
                output(output_base + 2 * k + 2) = im
            end do
            return
        end if

        even_base = input_base + n
        do i = 0, half_n - 1
            input(even_base + i + 1) = input(input_base + 2 * i + 1)
        end do
        even_fft_base = output_base + 2 * n
        call whisper_fft(input, even_base, half_n, output, even_fft_base, sin_table, cos_table)

        do i = 0, half_n - 1
            input(even_base + i + 1) = input(input_base + 2 * i + 2)
        end do
        odd_fft_base = even_fft_base + n
        call whisper_fft(input, even_base, half_n, output, odd_fft_base, sin_table, cos_table)

        step = WHISPER_N_FFT / n
        do k = 0, half_n - 1
            index = k * step
            re = cos_table(index + 1)
            im = -sin_table(index + 1)
            even_re = output(even_fft_base + 2 * k + 1)
            even_im = output(even_fft_base + 2 * k + 2)
            odd_re = output(odd_fft_base + 2 * k + 1)
            odd_im = output(odd_fft_base + 2 * k + 2)
            output(output_base + 2 * k + 1) = even_re + re * odd_re - im * odd_im
            output(output_base + 2 * k + 2) = even_im + re * odd_im + im * odd_re
            output(output_base + 2 * (k + half_n) + 1) = even_re - re * odd_re + im * odd_im
            output(output_base + 2 * (k + half_n) + 2) = even_im - re * odd_im - im * odd_re
        end do
    end subroutine whisper_fft

end module fortai_whisper_audio
