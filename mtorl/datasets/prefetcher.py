import torch


class PreFetcher:
    r"""
    This is a prefetcher accelerating copying of tensor from memory to
    GPU memory by calling the cuda() function in an non-blocking way.

    Note that this prefetcher takes a pytorch DataLoader as input.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.stream = torch.cuda.Stream()

        # built-in attrs
        self.next_batch = None

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch['image'] = self.next_batch['image'].cuda(non_blocking=True)
            self.next_batch['labels'] = self.next_batch['labels'].cuda(non_blocking=True)

    def __len__(self):
        return len(self.ori_loader)

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.next_batch
        if batch_data is not None:
            batch_data['image'].record_stream(torch.cuda.current_stream())
            for target in batch_data['labels']:
                target.record_stream(torch.cuda.current_stream())
        else:
            raise StopIteration
        self.preload()
        return batch_data
