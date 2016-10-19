from abc import ABCMeta, abstractmethod

class layer:
    __metaclass__ = ABCMeta
    
    def _p(self, prefix, name):
        """
        Get the name with prefix.
        """
        return '%s_%s' % (prefix, name)
