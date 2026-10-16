import re
from tokenizers import Tokenizer
import os
NT=os.environ.get('NT_DIR','/tmp/claude-0/nt');DL=NT+'/dl';W=NT+'/work'
LT=Tokenizer.from_file(f'{DL}/llama3_tok.json');QT=Tokenizer.from_file(f'{DL}/qwen_tok.json')
def nl(s):return len(LT.encode(s,add_special_tokens=False).ids)
def nq(s):return len(QT.encode(s,add_special_tokens=False).ids)
def tl(s):return (nl(' '+s),nq(' '+s))
NATG={'English':'UK','British':'UK','Scottish':'UK','Welsh':'UK','Northern':'UK','American':'US','Canadian':'CA','Australian':'AU','New':'NZ',
 'Irish':'IE','German':'DE','Austrian':'DE','French':'FR','Belgian':'BE','Italian':'IT','Spanish':'ES','Dutch':'NL','Danish':'DK','Swedish':'SE','Norwegian':'NO','Finnish':'FI',
 'Greek':'GR','Russian':'RU','Soviet':'RU','Polish':'PL','Czech':'CZ','Hungarian':'HU','Japanese':'JP','Argentine':'AR','Argentinian':'AR','Brazilian':'BR','Mexican':'MX','Portuguese':'PT','Swiss':'CH','Croatian':'HR','Serbian':'RS','Romanian':'RO','Ukrainian':'UA','Indian':'IN','Chinese':'CN','Turkish':'TR'}
