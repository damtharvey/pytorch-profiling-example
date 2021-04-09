import argparse
import torch
import torch2trt

import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, choices=models.MODELS)
    parser.add_argument('--ensemble-size', type=int, default=1)
    parser.add_argument('--input-shape', nargs='+', default=['3', '32', '32'])
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--to-tensorrt', action='store_true')
    parser.add_argument('--results-directory', type=str, default='results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_type = {16: torch.float16,
                 32: torch.float32,
                 64: torch.float64}[args.precision]

    model = models.homogeneous_ensemble(architecture=args.architecture,
                                        ensemble_size=args.ensemble_size,
                                        num_classes=args.classes).to(device).type(data_type)
    model.eval()

    inputs = (torch.randn((args.batch_size, *(int(x) for x in args.input_shape)),
                          dtype=data_type,
                          device=device),)
    if args.to_tensorrt:
        model_ = torch2trt.torch2trt(model, inputs, max_batch_size=args.batch_size, fp16_mode=(args.precision == 16))
        model_.eval()
    else:
        model_ = model

    trace_handler = torch.profiler.tensorboard_trace_handler(
        dir_name=args.results_directory,
        worker_name=f'{args.ensemble_size}x_{args.architecture}_'
                    f'{"tensorrt" if args.to_tensorrt else "pytorch"}_fp{args.precision}')

    with torch.profiler.profile(with_flops=True, on_trace_ready=trace_handler) as profiler:
        for _ in range(args.batches):
            _ = model_(*inputs)
            profiler.step()


if __name__ == '__main__':
    main()
