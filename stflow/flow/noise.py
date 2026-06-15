import torch


class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type
        self.device = torch.device(kwargs.get("device", "cpu"))

        if prior_sample_type == "gaussian":
            self.prior_sampler = self.gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = self.all_zeros
        elif prior_sample_type == "zinb":
            self.total_count = torch.as_tensor(
                kwargs.get("total_count", 1.0), dtype=torch.float32, device=self.device
            ).reshape(())
            self.logits = torch.as_tensor(
                kwargs.get("logits", 0.1), dtype=torch.float32, device=self.device
            ).reshape(())
            self.zi_logits = torch.as_tensor(
                kwargs.get("zi_logits", 0.0), dtype=torch.float32, device=self.device
            ).reshape(())
            self.prior_sampler = self.zinb_prior
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        return self.prior_sampler(shape)

    def gaussian_prior(self, shape):
        return torch.randn(shape, device=self.device)

    def all_zeros(self, shape):
        return torch.zeros(shape, device=self.device)

    def zinb_prior(self, shape):
        nb = torch.distributions.NegativeBinomial(
            total_count=self.total_count,
            logits=self.logits,
        )
        sample = nb.sample(shape).to(self.device)
        zero_prob = torch.sigmoid(self.zi_logits)
        zero_mask = torch.rand(shape, device=self.device) < zero_prob
        return sample.masked_fill(zero_mask, 0.0)
