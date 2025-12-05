import json
import os

class ScoreManager:
    def __init__(self, filename="highscores.json"):
        self.filename = filename
        self.max_scores = 5

    def load_scores(self):
        """Load high scores from file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def save_scores(self, scores):
        """Save high scores to file."""
        try:
            with open(self.filename, "w") as f:
                json.dump(scores, f, indent=2)
        except IOError:
            pass

    def add_score(self, score, coin_count):
        """Add a new score and return top scores."""
        scores = self.load_scores()
        scores.append({"score": int(score), "coins": int(coin_count)})
        scores.sort(key=lambda x: x["score"], reverse=True)
        scores = scores[:self.max_scores]
        self.save_scores(scores)
        return scores

    def get_high_scores(self):
        """Get the list of high scores."""
        return self.load_scores()
