import numpy as np
import os, sys
from WSFitFilt import WSTemplateFilt

templateDir = 'WSFiltTemplates'
templateFn = 'Hexis_FL3-template.txt'
rawWSFnList = ['Hexis_WS_FL3.txt']

mdd = os.environ['MKID_DATA_DIR']
for i,fn in enumerate(rawWSFnList):
    rawWSFnList[i] = os.path.join(mdd, fn)

wsFit = WSTemplateFilt()
wsFit.makeTemplate(rawWSFnList)
wsFit.saveTemplate(os.path.join(templateDir, templateFn))
