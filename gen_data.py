
import sys
import os
import json
import re


def main():
    directory = sys.argv[1]
    messages = []

    for chat in os.listdir(directory):
        path = directory + '/' + chat
        if not os.path.isdir(path):
            continue
        if not os.path.exists(path + '/message_1.json'):
            continue

        with open(path + '/message_1.json') as f:
            data = json.load(f)
            if len(data['participants']) > 2:
                continue
            for message in data['messages']:
                if 'content' not in message or message['content'][:7] == 'Reacted':
                    continue
                message['content'] = re.sub(r'[^A-Za-z0-9\ ]+', '', message['content'])
                if len(message['content']) == 0:
                    continue
                if message['content'].startswith('You can now message and call each other') or message['content'].startswith('You are now connected'):
                    continue
                if message['content'].startswith('You called') or message['content'].endswith('called you'):
                    continue
                if message['content'].startswith('You missed a call') or message['content'].endswith('missed your call'):
                    continue
                if 'http' in message['content']:
                    continue

                message['sender_name'] = 'WILL' if message['sender_name'] == 'Will Liang' else 'OTHER'
                message = message['sender_name'].upper() + '\n' + message['content'].lower()
                messages.append(message)

    print(f'{len(messages)} messages generated')
    with open('out.txt', 'w') as f:
        f.write('\n\n'.join(reversed(messages)))


if __name__ == '__main__':
    main()
