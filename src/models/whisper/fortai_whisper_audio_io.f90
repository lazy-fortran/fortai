module fortai_whisper_audio_io
    !! Small, dependency-free WAV reader for the native Whisper service.
    !!
    !! Network framing stays in the C transport; this module receives the
    !! already bounded byte string and turns PCM into the model's 16 kHz mono
    !! sample stream.  It deliberately accepts only uncompressed PCM16 WAV so
    !! unsupported codecs fail explicitly instead of silently producing bad
    !! transcripts.
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    use fortai_status, only: FORTAI_INVALID, FORTAI_IO_ERROR, FORTAI_UNSUPPORTED, status_t
    implicit none
    private

    integer(int64), parameter :: MAX_AUDIO_SAMPLES = 16000_int64 * 60_int64 * 30_int64

    public :: whisper_wav_decode
    public :: whisper_wav_read_file

contains

    integer(int64) function wav_byte(raw, position)
        character(len=*), intent(in) :: raw
        integer(int64), intent(in) :: position

        if (position < 1_int64 .or. position > int(len(raw), int64)) then
            wav_byte = -1_int64
        else
            wav_byte = iand(int(iachar(raw(int(position):int(position))), int64), 255_int64)
        end if
    end function wav_byte

    integer(int64) function wav_u16(raw, position)
        character(len=*), intent(in) :: raw
        integer(int64), intent(in) :: position
        integer(int64) :: first, second

        first = wav_byte(raw, position)
        second = wav_byte(raw, position + 1_int64)
        if (first < 0_int64 .or. second < 0_int64) then
            wav_u16 = -1_int64
        else
            wav_u16 = first + ishft(second, 8)
        end if
    end function wav_u16

    integer(int64) function wav_u32(raw, position)
        character(len=*), intent(in) :: raw
        integer(int64), intent(in) :: position
        integer(int64) :: first, second, third, fourth

        first = wav_byte(raw, position)
        second = wav_byte(raw, position + 1_int64)
        third = wav_byte(raw, position + 2_int64)
        fourth = wav_byte(raw, position + 3_int64)
        if (first < 0_int64 .or. second < 0_int64 .or. third < 0_int64 .or. fourth < 0_int64) then
            wav_u32 = -1_int64
        else
            wav_u32 = first + ishft(second, 8) + ishft(third, 16) + ishft(fourth, 24)
        end if
    end function wav_u32

    subroutine whisper_wav_decode(raw, samples, stat)
        character(len=*), intent(in) :: raw
        real(real32), allocatable, intent(out) :: samples(:)
        type(status_t), intent(out) :: stat
        integer(int64) :: position, chunk_size, chunk_end, fmt_end, data_start, data_size
        integer(int64) :: audio_format, channels, sample_rate, bits_per_sample, block_align
        integer(int64) :: frames, output_frames, frame, channel, sample_offset
        integer(int64) :: value, next_frame, source_frame
        real(real64) :: ratio, source_position, fraction, left, right, mixed
        real(real32), allocatable :: native(:)
        logical :: fmt_found, data_found

        call stat%clear()
        allocate(samples(0))
        if (len(raw) < 12) then
            call stat%set(FORTAI_INVALID, 'WAV payload is truncated')
            return
        end if
        if (raw(1:4) /= 'RIFF' .or. raw(9:12) /= 'WAVE') then
            call stat%set(FORTAI_INVALID, 'audio payload is not a RIFF/WAVE file')
            return
        end if

        position = 13_int64
        fmt_found = .false.
        data_found = .false.
        audio_format = 0_int64
        channels = 0_int64
        sample_rate = 0_int64
        bits_per_sample = 0_int64
        block_align = 0_int64
        data_start = 0_int64
        data_size = 0_int64
        do while (position + 7_int64 <= int(len(raw), int64))
            chunk_size = wav_u32(raw, position + 4_int64)
            if (chunk_size < 0_int64) exit
            chunk_end = position + 8_int64 + chunk_size
            if (chunk_end > int(len(raw), int64) + 1_int64) exit
            if (raw(int(position):int(position + 3_int64)) == 'fmt ') then
                if (chunk_size < 16_int64) then
                    call stat%set(FORTAI_INVALID, 'WAV fmt chunk is truncated')
                    return
                end if
                fmt_end = position + 8_int64 + chunk_size
                audio_format = wav_u16(raw, position + 8_int64)
                channels = wav_u16(raw, position + 10_int64)
                sample_rate = wav_u32(raw, position + 12_int64)
                block_align = wav_u16(raw, position + 20_int64)
                bits_per_sample = wav_u16(raw, position + 22_int64)
                if (fmt_end <= position) then
                    call stat%set(FORTAI_INVALID, 'WAV fmt chunk size is invalid')
                    return
                end if
                fmt_found = .true.
            else if (raw(int(position):int(position + 3_int64)) == 'data') then
                data_start = position + 8_int64
                data_size = chunk_size
                data_found = .true.
            end if
            position = chunk_end + mod(chunk_size, 2_int64)
            if (fmt_found .and. data_found) exit
        end do

        if (.not. fmt_found .or. .not. data_found) then
            call stat%set(FORTAI_INVALID, 'WAV file has no complete fmt/data chunks')
            return
        end if
        if (audio_format /= 1_int64 .or. bits_per_sample /= 16_int64) then
            call stat%set(FORTAI_UNSUPPORTED, 'only PCM16 WAV input is supported')
            return
        end if
        if (channels <= 0_int64 .or. channels > 32_int64 .or. sample_rate <= 0_int64 .or. &
            sample_rate > 384000_int64 .or. block_align /= 2_int64 * channels) then
            call stat%set(FORTAI_INVALID, 'WAV channel or sample-rate metadata is invalid')
            return
        end if
        if (data_size <= 0_int64 .or. mod(data_size, block_align) /= 0_int64) then
            call stat%set(FORTAI_INVALID, 'WAV PCM data is empty or unaligned')
            return
        end if
        frames = data_size / block_align
        if (frames > MAX_AUDIO_SAMPLES * 24_int64) then
            call stat%set(FORTAI_INVALID, 'WAV payload exceeds the native audio limit')
            return
        end if
        allocate(native(int(frames, int32)))
        do frame = 0_int64, frames - 1_int64
            mixed = 0.0_real64
            sample_offset = data_start + frame * block_align
            do channel = 0_int64, channels - 1_int64
                value = wav_u16(raw, sample_offset + 2_int64 * channel)
                if (value < 0_int64) then
                    deallocate(native)
                    call stat%set(FORTAI_INVALID, 'WAV PCM data is truncated')
                    return
                end if
                if (value >= 32768_int64) value = value - 65536_int64
                mixed = mixed + real(value, real64) / 32768.0_real64
            end do
            native(int(frame + 1_int64, int32)) = real(mixed / real(channels, real64), real32)
        end do

        if (sample_rate == 16000_int64) then
            call move_alloc(native, samples)
            return
        end if
        ratio = real(sample_rate, real64) / 16000.0_real64
        output_frames = max(1_int64, 1_int64 + int(real(frames - 1_int64, real64) / ratio, int64))
        if (output_frames > MAX_AUDIO_SAMPLES) then
            deallocate(native)
            call stat%set(FORTAI_INVALID, 'resampled audio exceeds the native audio limit')
            return
        end if
        allocate(samples(int(output_frames, int32)))
        do next_frame = 0_int64, output_frames - 1_int64
            source_position = real(next_frame, real64) * ratio
            source_frame = min(frames - 1_int64, int(source_position, int64))
            fraction = source_position - real(source_frame, real64)
            left = real(native(int(source_frame + 1_int64, int32)), real64)
            if (source_frame + 1_int64 < frames) then
                right = real(native(int(source_frame + 2_int64, int32)), real64)
            else
                right = left
            end if
            samples(int(next_frame + 1_int64, int32)) = real((1.0_real64 - fraction) * left + fraction * right, real32)
        end do
        deallocate(native)
    end subroutine whisper_wav_decode

    subroutine whisper_wav_read_file(path, samples, stat)
        character(len=*), intent(in) :: path
        real(real32), allocatable, intent(out) :: samples(:)
        type(status_t), intent(out) :: stat
        integer :: unit, ios
        integer(int64) :: file_size
        character(len=:), allocatable :: raw

        call stat%clear()
        allocate(samples(0))
        open(newunit=unit, file=trim(path), access='stream', form='unformatted', status='old', action='read', &
            iostat=ios)
        if (ios /= 0) then
            call stat%set(FORTAI_IO_ERROR, 'unable to open WAV file: ' // trim(path))
            return
        end if
        inquire(unit=unit, size=file_size)
        if (file_size <= 0_int64 .or. file_size > int(2_int64**31, int64)) then
            close(unit)
            call stat%set(FORTAI_INVALID, 'WAV file size is invalid')
            return
        end if
        allocate(character(len=int(file_size, int32)) :: raw)
        read(unit, iostat=ios) raw
        close(unit)
        if (ios /= 0) then
            deallocate(raw)
            call stat%set(FORTAI_IO_ERROR, 'unable to read WAV file: ' // trim(path))
            return
        end if
        call whisper_wav_decode(raw, samples, stat)
        deallocate(raw)
    end subroutine whisper_wav_read_file

end module fortai_whisper_audio_io
