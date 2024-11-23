from django.db import models

from django.db import models

class VideoComment(models.Model):
    video_id = models.CharField(max_length=100)
    comment = models.TextField()
    author = models.CharField(max_length=100)
    likes = models.IntegerField()
    reply_count = models.IntegerField()

    def __str__(self):
        return f"Comment by {self.author} on video {self.video_id}"
