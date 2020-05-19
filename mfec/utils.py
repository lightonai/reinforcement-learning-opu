class Utils:
    def __init__(self, frames_per_epoch, max_frames):
        self.frames_per_epoch = frames_per_epoch
        self.max_frames = max_frames
        self.total_frames = 0
        self.epoch = 1
        self.epoch_episodes = 0
        self.epoch_frames = 0
        self.epoch_reward_sum = 0
        self.epoch_reward_max = 0

    def end_episode(self, episode_frames, episode_reward):
        self.epoch_episodes += 1
        self.epoch_frames += episode_frames
        self.epoch_reward_sum += episode_reward

        if episode_reward > self.epoch_reward_max:
            self.epoch_reward_max = episode_reward
        self.total_frames += episode_frames

        message = "Epoch: {}\tEpisode: {}\tReward: {}\tEpoch frames: {}/{}"
        results = [
            self.epoch,
            self.epoch_episodes,
            int(episode_reward),
            self.epoch_frames,
            self.frames_per_epoch,
        ]
        print(message.format(*results))

    def end_epoch(self):
        results = [
            self.epoch,
            self.epoch_episodes,
            self.epoch_frames,
            int(self.epoch_reward_sum),
            round(self.epoch_reward_sum / self.epoch_episodes),
            int(self.epoch_reward_max),
        ]
        # self.results_file.write("{},{},{},{},{},{}\n".format(*results))
        # self.results_file.flush()

        message = (
            "\nEpoch: {}\tEpisodes: {}\tFrames: {}\tReward-Sum: {}\t"
            "Reward-Avg: {}\tReward-Max: {}\tTotal-Frames: {}/{}\n"
        )
        results = results + [self.total_frames, self.max_frames]
        print(message.format(*results))

        self.epoch += 1
        self.epoch_episodes = 0
        self.epoch_frames = 0
        self.epoch_reward_sum = 0
        self.epoch_reward_max = 0

    def close(self):
        # self.results_file.close()
        pass
