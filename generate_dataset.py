import pandas as pd
import praw
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

reddit = praw.Reddit(
    client_id="Zj7P6jW7euSq6kOliBIwjw",
    client_secret="9vgQbIUcwgFHV6sh155F7FQwcaFcCw",
    user_agent="testscript middelz2",
    username="middelz2",
)


def generate_dataset():
    post_selection = pd.read_csv("dataset_posts_selection.csv")
    final_dataset = pd.DataFrame(
        columns=['url', 'topic', 'comment_id', 'author', 'comment', 'created_utc', 'ups', 'replies'])
    for index, row in post_selection.iterrows():
        submission = reddit.submission(url=row['url'])
        submission.comments.replace_more(limit=0)
        all_comments = submission.comments.list()
        for comment in all_comments:
            if comment.replies._comments:
                if len(comment.replies._comments) > 0:
                    reply_ids = []
                    for reply_id in comment.replies._comments:
                        reply_ids.append(reply_id)
                    reply_ids = str(reply_ids)
                else:
                    reply_ids = str([])

                comment_preprocessed = re.sub('[^\S ]+', '', str(comment.body).strip())

                new_row = {'url': str(row["url"]), 'topic': str(row["topic"]), 'comment_id': comment.id, 'author': comment.author, 'comment': comment_preprocessed,
                           'created_utc': comment.created_utc, 'ups': comment.ups, 'replies': reply_ids}
                final_dataset = final_dataset.append(new_row, ignore_index=True)

        final_dataset.to_csv("dataset_raw.csv")

generate_dataset()
