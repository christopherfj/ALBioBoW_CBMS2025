import ast
import sys

FILENAME = sys.argv[1]
min_idx = int( sys.argv[2] )
max_idx = int( sys.argv[3] )
MODELS = [sys.argv[4]]
CURVES = [sys.argv[5]]


import warnings
warnings.filterwarnings("ignore")
import logging
logging.captureWarnings(True)
logging.disable(sys.maxsize)
from cregex import *
from utils import *
from curves import *
seed_everything()

N_CLASSES = {'FUMADOR':2, 'OBESIDAD':2, 'OBESIDAD_TIPOS':3}[FILENAME]
HYPERPARAMS['bert']['n_classes'] = N_CLASSES
HYPERPARAMS['setfit']['n_classes'] = N_CLASSES

create_paths(FILENAME)
    
with open( os.path.join( os.getcwd(), 'snippets_procesados_'+FILENAME),  'rb') as a:
    data = pickle.load(a)    
    data = sorted(data, key = lambda x:x[0], reverse = False)
    DATA = np.array( [snippet for snippet, classe in data] )#[:300]
    CLASSES = np.array( [classe for snippet, classe in data])#[:300]

print(FILENAME)
RUNS = 1
FOLDS = 5
folds = KFold(n_splits = FOLDS, shuffle = False)
idxs = np.arange(0, len(DATA))    

for r in range(RUNS):
    idxs = shuffle(idxs, random_state = SEED)
    CLASSES = CLASSES[idxs]
    DATA = DATA[idxs]    
    k = -1
    for train_index, test_index in folds.split(idxs):
        k+=1    
        print('fold:', k+1)
        if (k+1) not in list(range(min_idx, max_idx+1)):
            continue
            
        for CURVE in CURVES:
            
            for MODEL in MODELS:
                
                
                if 'PL' in CURVE:
                    if 'regex' in MODEL or 'bert' in MODEL or 'setfit' in MODEL:
                        continue
                    
                print('CURVE:', CURVE)
                print('MODEL:', MODEL)

                X_train = copy.deepcopy( DATA[train_index] )
                y_train = copy.deepcopy( CLASSES[train_index] )
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
                X_test = copy.deepcopy( DATA[test_index] )
                y_test = copy.deepcopy( CLASSES[test_index] )   

                curve = Curves(
                    X_train, y_train,
                    X_val, y_val,
                    X_test,
                    N_CLASSES, CURVE, MODEL,
                    BATCH, FILENAME
                )
                curve.learningCurve()
                results = [ 
                           curve.results['scores'], 
                           curve.results['x'], 
                           curve.results['y'], 
                           curve.results['y_u_dst'], 
                           curve.results['y_clf'], 
                           curve.results['dst_cregex'], 
                           curve.HYPERPARAMS,        #new
                           curve.N_FEATURES,         #new 
                           y_test
                ]
                with open( os.path.join( os.getcwd(), 'out', 'RESULTSLC', CURVE, FILENAME, FILENAME+'_'+MODEL+'_'+CURVE+'_k'+str(k+1)+'.pkl' ), 'wb' ) as a:
                    pickle.dump(results, a, protocol=2)

                del X_train
                del X_val
                del X_test
                del y_train
                del y_val
                del y_test
                gc.collect()

