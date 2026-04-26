import sys
msg = sys.stdin.read()
cleaned = '\n'.join(l for l in msg.splitlines() if 'Made-with' not in l).strip()
print(cleaned)
