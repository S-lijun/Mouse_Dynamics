import os
import re
from collections import defaultdict

ROOT = "Data/DFL-dataset"


def restore_sessions():
    for user in os.listdir(ROOT):
        user_dir = os.path.join(ROOT, user)

        if not os.path.isdir(user_dir):
            continue

        # 收集该 user 目录内所有 part 文件
        part_files = [
            f for f in os.listdir(user_dir)
            if re.search(r"_part\d+\.CSV$", f)
        ]

        # 按 session 前缀分组
        sessions = defaultdict(list)

        for filename in part_files:
            match = re.match(r"(.+)_part(\d+)\.CSV$", filename)
            if match:
                session_prefix = match.group(1)
                part_index = int(match.group(2))
                sessions[session_prefix].append((part_index, filename))

        # 对每个 session 进行恢复
        for session_prefix, parts in sessions.items():
            output_path = os.path.join(user_dir, f"{session_prefix}.CSV")

            # 如果已经恢复过就跳过
            if os.path.exists(output_path):
                print(f"Already restored: {output_path}")
                continue

            print(f"Restoring session: {session_prefix} (User: {user})")

            # 按数值排序
            parts.sort(key=lambda x: x[0])

            with open(output_path, "w", encoding="utf-8") as outfile:
                for i, (_, filename) in enumerate(parts):
                    part_path = os.path.join(user_dir, filename)

                    with open(part_path, "r", encoding="utf-8") as infile:
                        if i == 0:
                            # 第一段写完整内容
                            outfile.write(infile.read())
                        else:
                            # 之后跳过 header
                            next(infile)
                            outfile.write(infile.read())

            print(f"  -> Created {output_path}")


if __name__ == "__main__":
    restore_sessions()