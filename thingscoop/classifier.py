import cPickle
#import caffe
import cv2
import glob
import logging
import numpy
import os
from dd_client import DD

class ImageClassifier(object):
    def __init__(self, model, gpu_mode=False):
        self.model = model
        
        kwargs = {}

        if self.model.get("image_dims"):
            kwargs['image_dims'] = tuple(self.model.get("image_dims"))

        if self.model.get("channel_swap"):
            kwargs['channel_swap'] = tuple(self.model.get("channel_swap"))

        if self.model.get("raw_scale"):
            kwargs['raw_scale'] = float(self.model.get("raw_scale"))

        if self.model.get("mean"):
            kwargs['mean'] = numpy.array(self.model.get("mean"))
        
        #self.net = caffe.Classifier(
        #    model.deploy_path(),
        #    model.model_path(),
        #    **kwargs
        #)
        
        # dd variables
        self.host = 'localhost'
        self.sname = 'places'
        description = 'places image prediction'
        self.mllib = 'caffe'
        self.dd = DD(self.host)
        self.dd.set_return_format(self.dd.RETURN_PYTHON)
        dd_model = {'repository':model.model_dir}
        print 'dd_model=',dd_model
        parameters_input = {'connector':'image'}
        parameters_mllib = {'nclasses':int(model.info['classes'])}#1000} #TODO: from yaml file in model object
        parameters_output = {}
        self.dd.delete_service(self.sname)
        self.dd.put_service(self.sname,dd_model,description,self.mllib,
                            parameters_input,parameters_mllib,parameters_output)
        
        self.confidence_threshold = 0.1
        
        #if gpu_mode:
        #    caffe.set_mode_gpu()
        #else:
        #    caffe.set_mode_cpu()

        #self.labels = numpy.array(model.labels())

        #if self.model.bet_path():
        #    self.bet = cPickle.load(open(self.model.bet_path()))
        #    self.bet['words'] = map(lambda w: w.replace(' ', '_'), self.bet['words'])
        #else:
        self.bet = None
        
        #self.net.forward()

    def classify_image(self, filename):
        image = caffe.io.load_image(open(filename))
        scores = self.net.predict([image], oversample=True).flatten()

        if self.bet:
            expected_infogain = numpy.dot(self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']
            infogain_sort = expected_infogain.argsort()[::-1]
            results = [
                (self.bet['words'][v], float(expected_infogain[v]))
                for v in infogain_sort
                if expected_infogain[v] > self.confidence_threshold
            ]

        else:
            indices = (-scores).argsort()
            predictions = self.labels[indices]
            results = [
                (p, float(scores[i]))
                for i, p in zip(indices, predictions)
                if scores[i] > self.confidence_threshold
            ]

        return results

    def classify_image_dd(self, filename):
        #TODO:
        #- predict
        data = [filename]
        parameters_input = {'width':224,'height':224}
        parameters_mllib = {'gpu':False}
        parameters_output = {'best':3}
        predict_output = self.dd.post_predict(self.sname,data,parameters_input,parameters_mllib,parameters_output)
        #TODO: error handling
        
        #- turn results into the proper format
        print 'predict_output=',predict_output
        results = []
        for p in predict_output['body']['predictions']['classes']:
            results.append((p['cat'],p['prob']))
            #results.append((int(p['cat']),p['prob']))
        return results
