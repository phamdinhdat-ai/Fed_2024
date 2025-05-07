from transformers import TrainerCallback

class SaveBestTrainingLossCallback(TrainerCallback):
    def __init__(self, save_steps, output_dir):
        self.output_dir = output_dir
        self.save_steps = save_steps
        
        self._actual_save_steps = None
        self._warmup_steps = 0
        
        self.best_loss = [float('inf')]

    def on_train_begin(self, args, state, control, **kwargs):
        self._warmup_steps = int(args.warmup_ratio * state.max_steps)
        print(f"[AUTO_SAVE] IGNORE {self._warmup_steps} STEPS AS WARMUP")
        
        total_steps = state.max_steps
        if isinstance(self.save_steps, float) and 0 < self.save_steps < 1:
            self._actual_save_steps = max(1, int(total_steps * self.save_steps))
        elif isinstance(self.save_steps, int) and self.save_steps > 0:
            self._actual_save_steps = self.save_steps
        else:
            raise ValueError("save_steps must be a positive int or a float between 0 and 1.")

        print(f"[AUTO_SAVE] SAVE EACH {self._actual_save_steps} STEPS")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Ensure logs are available and contain 'loss'
        if logs is None or 'loss' not in logs:
            return

        current_loss = logs['loss']
        global_step = state.global_step

        if global_step < self._warmup_steps:
            return

        if self._actual_save_steps and global_step % self._actual_save_steps != 0:
            return
            
        if current_loss > self.best_loss[-1]:
            return
            
        self.best_loss.append(current_loss)
        
        print(f"[AUTO_SAVE] Step: {global_step: >5} | New best training loss: {self.best_loss[-2]:.4f} -> {self.best_loss[-1]:.4f}.")
        
        model = kwargs['model']
        
        model.save_pretrained(
            self.output_dir, safe_serialization=True
        )
        model.config.save_pretrained(
            self.output_dir
        )

        wandb.log({
            "best_loss": self.best_loss[-1],
            "best_step": global_step
        })
# Đại DU
# Cho mấy az hàm tracking trainning los để save chứ lone Trainer support tracking trên mỗi tập eval à :v
# Đại DU
# 🌜
