import tensorflow as tf
import re

def exportModel(sess, filename, filters=['.*']):
    def matchName(name):
        for f in filters:
            if isinstance(f, str):
                if re.match("^"+f+"$", name):
                    return True
            else:
                if f(name):
                    return True
                    
        return False

    vars = []
    print("Exporting...")
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        if matchName(v.op.name):
            print("   "+v.op.name)
            vars.append(v)
    print("Done.")
        
    saver = tf.train.Saver(var_list=vars)
    saver.save(sess, filename, write_meta_graph=False)