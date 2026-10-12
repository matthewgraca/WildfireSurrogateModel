from wildfire_simulator.scheduled_sampler import ScheduledSampler

def test_scheduled_sampler():
    sampler = ScheduledSampler(k=0.1, t0=40)
    assert int(100 * sampler.get_prob(23)) == 15

