import os
from matchedfilt import WSTemplateFilt

raise RuntimeError('Talk to Jeb about restructuring this file')

templateDir = 'templates'
templateFn = 'Hexis_FL3-template.txt'
rawWSFnList = ['Hexis_WS_FL3.txt']

mdd = os.environ['MKID_DATA_DIR']
for i,fn in enumerate(rawWSFnList):
    rawWSFnList[i] = os.path.join(mdd, fn)

wsFit = WSTemplateFilt()
wsFit.makeTemplate(rawWSFnList)
wsFit.saveTemplate(os.path.join(templateDir, templateFn))

