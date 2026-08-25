program test_fortai_smoke
    use, intrinsic :: iso_c_binding, only: c_float, c_int16_t, c_int64_t
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
    use fortai_arena, only: arena_t
    use fortai_backend_cpu, only: cpu_matvec, cpu_matvec_inplace
    use fortai_cache, only: cache_path, pack_key_t
    use fortai_device, only: device_cpu, device_t
    use fortai_gguf, only: gguf_validate_header
    use fortai_gguf_runtime, only: GGML_TYPE_Q8_0, gguf_tensor_t
    use fortai_model_ir, only: model_ir_t
    use fortai_plan_ir, only: plan_ir_t
    use fortai_qwen35, only: qwen35_config_t
    use fortai_qwen35_dflash2, only: qwen35_dflash2_available
    use fortai_qwen35_gdn, only: gdn_reference_step
    use fortai_qwen35_mtp, only: qwen35_mtp_available
    use fortai_qwen35_vision, only: qwen35_vision_available
    use fortai_sampler, only: sampler_t
    use fortai_speculative, only: speculative_t
    use fortai_status, only: status_t
    use fortai_tensor, only: tensor_t
    use fortai_tokenizer, only: tokenizer_t
    implicit none

    interface
        function fortai_float_to_half_test(value) bind(C, name='fortai_float_to_half') result(bits)
            import c_float, c_int16_t
            real(c_float), value, intent(in) :: value
            integer(c_int16_t) :: bits
        end function fortai_float_to_half_test

        subroutine fortai_gdn_step_test(state, key, value, query, decay, beta, head_size, &
                output_scale, output) bind(C, name='fortai_gdn_step')
            import c_float, c_int64_t
            real(c_float), intent(inout) :: state(*)
            real(c_float), intent(in) :: key(*), value(*), query(*)
            real(c_float), value, intent(in) :: decay, beta, output_scale
            integer(c_int64_t), value, intent(in) :: head_size
            real(c_float), intent(out) :: output(*)
        end subroutine fortai_gdn_step_test

        subroutine fortai_flash_attention_f16_test(query, key_cache, value_cache, count, &
                key_stride, value_stride, key_size, value_size, scale, output) &
                bind(C, name='fortai_flash_attention_f16')
            import c_float, c_int64_t
            real(c_float), intent(in) :: query(*), key_cache(*), value_cache(*)
            integer(c_int64_t), value, intent(in) :: count, key_stride, value_stride
            integer(c_int64_t), value, intent(in) :: key_size, value_size
            real(c_float), value, intent(in) :: scale
            real(c_float), intent(out) :: output(*)
        end subroutine fortai_flash_attention_f16_test

        subroutine fortai_silu_product(left, right, count) bind(C, name='fortai_silu_product')
            import c_float, c_int64_t
            real(c_float), intent(inout) :: left(*)
            real(c_float), intent(in) :: right(*)
            integer(c_int64_t), value, intent(in) :: count
        end subroutine fortai_silu_product
    end interface

    integer :: failures

    failures = 0
    call test_tensor(failures)
    call test_cpu_matvec(failures)
    call test_tokenizer(failures)
    call test_sampler(failures)
    call test_speculative(failures)
    call test_ir_contracts(failures)
    call test_device_and_arena(failures)
    call test_gguf_cache_qwen(failures)
    call test_gguf_q8_matvec(failures)
    call test_half_conversion(failures)
    call test_flash_attention_kernel(failures)
    call test_gdn_kernel(failures)
    call test_silu_kernel(failures)
    if (failures > 0) error stop 1
    print '(a)', 'FortAI smoke tests passed'

contains

    subroutine require(condition, message, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: message
        integer, intent(inout) :: failures

        if (.not. condition) then
            print '(a)', 'FAIL: ' // trim(message)
            failures = failures + 1
        end if
    end subroutine require

    subroutine test_tensor(failures)
        integer, intent(inout) :: failures
        type(tensor_t) :: tensor
        type(status_t) :: stat

        call tensor%init([2_int32, 3_int32], stat)
        call require(stat%is_ok(), 'tensor initialization', failures)
        call require(tensor%numel() == 6_int64, 'tensor element count', failures)
        call tensor%fill(2.5_real64)
        call require(abs(tensor%value_at(4_int32, stat) - 2.5_real64) < 1.0e-12_real64, &
            'tensor fill and access', failures)
        call require(stat%is_ok(), 'tensor access status', failures)
    end subroutine test_tensor

    subroutine test_cpu_matvec(failures)
        integer, intent(inout) :: failures
        real(real64) :: matrix(2, 2), vector(2)
        real(real64) :: inplace_result(2)
        real(real64), allocatable :: result(:)
        type(status_t) :: stat

        matrix = reshape([1.0_real64, 3.0_real64, 2.0_real64, 4.0_real64], [2, 2])
        vector = [5.0_real64, 6.0_real64]
        call cpu_matvec(matrix, vector, result, stat)
        call require(stat%is_ok(), 'CPU matvec status', failures)
        call require(size(result) == 2, 'CPU matvec result shape', failures)
        call require(abs(result(1) - 17.0_real64) < 1.0e-12_real64, &
            'CPU matvec first row', failures)
        call require(abs(result(2) - 39.0_real64) < 1.0e-12_real64, &
            'CPU matvec second row', failures)
        call cpu_matvec_inplace(matrix, vector, inplace_result, stat)
        call require(stat%is_ok(), 'in-place CPU matvec status', failures)
        call require(maxval(abs(inplace_result - result)) < 1.0e-12_real64, &
            'in-place CPU matvec agrees with allocating path', failures)
    end subroutine test_cpu_matvec

    subroutine test_tokenizer(failures)
        integer, intent(inout) :: failures
        type(tokenizer_t) :: tokenizer
        integer(int32), allocatable :: ids(:)
        character(len=:), allocatable :: decoded

        call tokenizer%encode('hello world', ids)
        decoded = tokenizer%decode(ids)
        call require(decoded == 'hello world', 'byte tokenizer round trip', failures)
        call require(ids(1) == 1_int32, 'tokenizer BOS token', failures)
        call require(ids(size(ids)) == 2_int32, 'tokenizer EOS token', failures)
    end subroutine test_tokenizer

    subroutine test_sampler(failures)
        integer, intent(inout) :: failures
        type(sampler_t) :: sampler
        type(status_t) :: stat
        integer(int32) :: token_id

        call sampler%greedy([0.2_real64, 1.5_real64, 1.4_real64], token_id, stat)
        call require(stat%is_ok(), 'greedy sampler status', failures)
        call require(token_id == 2_int32, 'greedy sampler argmax', failures)
    end subroutine test_sampler

    subroutine test_speculative(failures)
        integer, intent(inout) :: failures
        type(speculative_t) :: speculative
        type(status_t) :: stat
        integer(int32) :: accepted

        call speculative%accept_prefix([10_int32, 11_int32, 12_int32], &
            [10_int32, 11_int32, 99_int32], accepted, stat)
        call require(stat%is_ok(), 'speculative prefix status', failures)
        call require(accepted == 2_int32, 'speculative accepted prefix', failures)
    end subroutine test_speculative

    subroutine test_ir_contracts(failures)
        integer, intent(inout) :: failures
        type(model_ir_t) :: model
        type(plan_ir_t) :: plan
        type(status_t) :: stat

        call model%init('qwen3.5', 1024_int32, 64_int32, 2_int32, 4096_int32, &
            'Q4_K_M', stat)
        call require(stat%is_ok(), 'ModelIR validation', failures)
        call plan%init()
        call plan%append('decode', 'cpu_matvec', 1_int32, stat)
        call require(stat%is_ok(), 'PlanIR append', failures)
        call plan%validate(stat)
        call require(stat%is_ok(), 'PlanIR validation', failures)
        call require(plan%count() == 1_int32, 'PlanIR step count', failures)
    end subroutine test_ir_contracts

    subroutine test_device_and_arena(failures)
        integer, intent(inout) :: failures
        type(arena_t) :: arena
        type(device_t) :: device
        type(status_t) :: stat

        device = device_cpu()
        call require(device%is_supported(), 'CPU device support', failures)
        call require(device%label() == 'host-cpu', 'CPU device label', failures)
        call arena%init(64_int64, stat)
        call require(stat%is_ok(), 'arena initialization', failures)
        call arena%reserve(16_int64, stat)
        call require(stat%is_ok(), 'arena reservation', failures)
        call require(arena%available() == 48_int64, 'arena accounting', failures)
    end subroutine test_device_and_arena

    subroutine test_gguf_cache_qwen(failures)
        integer, intent(inout) :: failures
        real(real64) :: state(2), input(2), gate(2)
        real(real64), allocatable :: output(:)
        type(pack_key_t) :: key
        type(qwen35_config_t) :: config
        type(status_t) :: stat
        character(len=:), allocatable :: path

        call gguf_validate_header('GGUF', 3_int32, stat)
        call require(stat%is_ok(), 'GGUF header validation', failures)
        key%model = 'qwen35'
        key%backend = 'cuda'
        key%variant = 'sm86'
        path = cache_path('/tmp/fortai', key)
        call require(path == '/tmp/fortai/qwen35/cuda-sm86-abi1.pack', &
            'packed cache path', failures)
        call config%init(64_int32, 4_int32, 2_int32, 2_int32, stat)
        call require(stat%is_ok(), 'Qwen configuration validation', failures)
        state = [1.0_real64, 2.0_real64]
        input = [3.0_real64, 4.0_real64]
        gate = [1.0_real64, 2.0_real64]
        call gdn_reference_step(state, input, 0.5_real64, gate, output, stat)
        call require(stat%is_ok(), 'GDN reference status', failures)
        call require(abs(output(1) - 3.5_real64) < 1.0e-12_real64, &
            'GDN first output', failures)
        call require(abs(output(2) - 10.0_real64) < 1.0e-12_real64, &
            'GDN second output', failures)
        call require(qwen35_mtp_available(), 'native MTP implementation is compiled', failures)
        call require(.not. qwen35_dflash2_available(), &
            'DFlash2 is explicit pending work', failures)
        call require(.not. qwen35_vision_available(), &
            'vision is explicit pending work', failures)
    end subroutine test_gguf_cache_qwen

    subroutine test_gguf_q8_matvec(failures)
        integer, intent(inout) :: failures
        type(gguf_tensor_t) :: tensor, second, third
        type(status_t) :: stat
        integer(int8) :: quantized(34)
        real(real32) :: scales(1), vector(32), output(1), second_output(2), third_output(1), row_values(32)
        integer :: i, row

        tensor%value_type = GGML_TYPE_Q8_0
        tensor%shape = [32_int64, 1_int64]
        allocate (tensor%bytes(34))
        tensor%bytes = 0_int8
        tensor%bytes(2) = int(z'3c', int8)
        do i = 1, 32
            tensor%bytes(i + 2) = int(i, int8)
        end do
        vector = 1.0_real32
        call tensor%matvec_q8(vector, output, quantized, scales, stat)
        call require(stat%is_ok(), 'Q8 activation matvec status', failures)
        ! Q8_0 activation scales are stored as FP16, matching llama.cpp.
        call require(abs(real(output(1), real64) - 527.9677734375_real64) < 1.0e-4_real64, &
            'Q8 activation matvec independent oracle', failures)
        call tensor%get_row(1_int64, row_values, stat)
        call require(stat%is_ok(), 'Q8 row dequantization status', failures)
        call require(maxval(abs(real(row_values, real64) - real([(i, i = 1, 32)], real64))) < &
            1.0e-5_real64, 'Q8 row dequantization independent oracle', failures)

        second%value_type = GGML_TYPE_Q8_0
        second%shape = [32_int64, 2_int64]
        allocate (second%bytes(68))
        second%bytes = 0_int8
        third%value_type = GGML_TYPE_Q8_0
        third%shape = [32_int64, 1_int64]
        allocate (third%bytes(34))
        third%bytes = 0_int8
        do row = 0, 1
            second%bytes(row * 34 + 1) = int(z'00', int8)
            second%bytes(row * 34 + 2) = int(z'3c', int8)
            do i = 1, 32
                second%bytes(row * 34 + i + 2) = int((row + 1) * i, int8)
            end do
        end do
        third%bytes(2) = int(z'3c', int8)
        do i = 1, 32
            third%bytes(i + 2) = int(3 * i, int8)
        end do
        call tensor%matvec_triplet_q8(second, third, vector, output, second_output, &
            third_output, quantized, scales, stat)
        call require(stat%is_ok(), 'Q8 triplet activation matvec status', failures)
        call require(maxval(abs(real([output(1), second_output, third_output], real64) - &
            [527.9677734375_real64, 527.9677734375_real64, 1055.935546875_real64, &
            1583.9033203125_real64])) < 1.0e-4_real64, &
            'Q8 triplet activation matvec independent oracle', failures)
    end subroutine test_gguf_q8_matvec

    subroutine test_half_conversion(failures)
        integer, intent(inout) :: failures

        call require(fortai_float_to_half_test(1.0_real32) == int(z'3c00', c_int16_t), &
            'F16 exact value conversion', failures)
        call require(fortai_float_to_half_test(1.00048828125_real32) == int(z'3c00', c_int16_t), &
            'F16 midpoint rounds to even lower value', failures)
        call require(fortai_float_to_half_test(1.00146484375_real32) == int(z'3c02', c_int16_t), &
            'F16 midpoint rounds to even upper value', failures)
        call require(fortai_float_to_half_test(2.98023223876953125e-8_real32) == &
            int(z'0000', c_int16_t), 'F16 subnormal midpoint rounds to zero', failures)
        call require(fortai_float_to_half_test(8.94069671630859375e-8_real32) == &
            int(z'0002', c_int16_t), 'F16 subnormal midpoint rounds to even value', failures)
    end subroutine test_half_conversion

    subroutine test_flash_attention_kernel(failures)
        integer, intent(inout) :: failures
        integer, parameter :: key_size = 32, value_size = 32, key_count = 3
        integer, parameter :: key_stride = 35, value_stride = 37
        real(real32), parameter :: query_pattern(8) = [ &
            1.00031_real32, -0.33327_real32, 0.20007_real32, -0.14291_real32, &
            0.09094_real32, -0.07689_real32, 0.05887_real32, -0.05261_real32]
        real(real32), parameter :: key_pattern(8, key_count) = reshape([ &
            0.0_real32, 0.0_real32, 0.0_real32, 0.0_real32, &
            0.0_real32, 0.0_real32, 0.0_real32, 0.0_real32, &
            -0.75_real32, 0.5_real32, -1.25_real32, 0.25_real32, &
            -0.375_real32, 0.875_real32, -0.625_real32, 0.125_real32, &
            0.5_real32, -1.0_real32, 0.75_real32, -0.5_real32, &
            1.25_real32, -0.25_real32, 0.375_real32, -0.875_real32], [8, key_count])
        real(real32), parameter :: value_pattern(8, key_count) = reshape([ &
            1.0_real32, -0.75_real32, 0.333251953125_real32, 7.5_real32, &
            -3.125_real32, 0.015625_real32, 19.0_real32, -0.0625_real32, &
            -0.8125_real32, 1.125_real32, -2.5_real32, 0.0625_real32, &
            8.25_real32, -0.5_real32, -11.0_real32, 0.333251953125_real32, &
            2.25_real32, -1.375_real32, 4.5_real32, -6.25_real32, &
            0.125_real32, 3.75_real32, 0.03125_real32, -8.0_real32], [8, key_count])
        real(real32), parameter :: expected_pattern(8) = [ &
            2.02924752_real32, -1.24691701_real32, 3.82288337_real32, -4.31401777_real32, &
            -0.158871993_real32, 3.17909908_real32, 2.34450269_real32, -6.79623699_real32]
        real(real32) :: query(key_size), key_cache(key_stride, key_count)
        real(real32) :: value_cache(value_stride, key_count), output(value_size)
        integer :: block, key_index

        key_cache = 123.5_real32
        value_cache = -321.0_real32
        do block = 0, 3
            query(block * 8 + 1:block * 8 + 8) = query_pattern
            do key_index = 1, key_count
                key_cache(block * 8 + 1:block * 8 + 8, key_index) = &
                    key_pattern(:, key_index)
                value_cache(block * 8 + 1:block * 8 + 8, key_index) = &
                    value_pattern(:, key_index)
            end do
        end do

        call fortai_flash_attention_f16_test(query, key_cache, value_cache, &
            int(key_count, c_int64_t), int(key_stride, c_int64_t), &
            int(value_stride, c_int64_t), int(key_size, c_int64_t), &
            int(value_size, c_int64_t), 0.37_real32, output)
        ! Independent expected values from llama.cpp b10566's CPU graph. The
        ! non-unit strides and alternating score maxima exercise cache layout,
        ! online-softmax rescaling, and the FP16 value accumulator.
        do block = 0, 3
            call require(maxval(abs(output(block * 8 + 1:block * 8 + 8) - &
                expected_pattern)) < 1.0e-5_real32, &
                'F16 flash-attention independent oracle', failures)
        end do
    end subroutine test_flash_attention_kernel

    subroutine test_gdn_kernel(failures)
        integer, intent(inout) :: failures
        real(real32) :: state(4), key(2), value(2), query(2), output(2)

        state = [1.0_real32, 2.0_real32, 3.0_real32, 4.0_real32]
        key = [1.0_real32, 2.0_real32]
        value = [5.0_real32, 6.0_real32]
        query = [2.0_real32, 1.0_real32]
        call fortai_gdn_step_test(state, key, value, query, 0.5_real32, 0.25_real32, &
            int(2, c_int64_t), 1.0_real32 / sqrt(2.0_real32), output)
        call require(maxval(abs(output - [4.5_real32, 5.5_real32] / sqrt(2.0_real32))) < 1.0e-5, &
            'fused GDN kernel independent oracle', failures)
        call require(maxval(abs(state - [1.125_real32, 2.25_real32, 1.625_real32, 2.25_real32])) < 1.0e-5, &
            'fused GDN state update', failures)
    end subroutine test_gdn_kernel

    subroutine test_silu_kernel(failures)
        integer, intent(inout) :: failures
        real(real32) :: left(256), right(256), expected(256)
        integer :: i

        do i = 1, size(left)
            left(i) = -20.0_real32 + 40.0_real32 * real(i - 1, real32) / &
                real(size(left) - 1, real32)
            right(i) = 1.0_real32 + 0.25_real32 * sin(real(i, real32))
            expected(i) = left(i) / (1.0_real32 + exp(-left(i))) * right(i)
        end do
        call fortai_silu_product(left, right, int(size(left), c_int64_t))
        if (maxval(abs(left - expected)) > 2.0e-5_real32) then
            failures = failures + 1
            write (*, '(a, es12.4)') 'silu kernel mismatch: ', maxval(abs(left - expected))
        end if
    end subroutine test_silu_kernel

end program test_fortai_smoke
