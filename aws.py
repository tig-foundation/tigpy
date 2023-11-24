import boto3


class Clients:
    _kwargs = {}

    @classmethod
    def setup(cls, **kwargs):
        cls._kwargs = kwargs
        
    @classmethod
    def get(cls, name):
        if not hasattr(cls, name):
            setattr(cls, name, boto3.client(name, **cls._kwargs))
        return getattr(cls, name)
  
