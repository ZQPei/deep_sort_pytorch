from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(args, use_cuda):
    return DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
    