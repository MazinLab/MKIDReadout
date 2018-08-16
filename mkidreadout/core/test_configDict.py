import unittest
from unittest import TestCase
import ruamel.yaml
from mkidreadout.core.config import ConfigDict, RESERVED, defaultconfigfile

yaml = ruamel.yaml.YAML()

class TestConfigDict(TestCase):
    def test_to_yaml(self):
        """Does not test syntax, only function and not well"""
        with open(defaultconfigfile(),'r') as f:
            cd = yaml.load(f)
        out = cd.dump()
        self.assertIsInstance(out, str)

    def test_from_yaml(self):
        with open(defaultconfigfile(),'r') as f:
            cd = yaml.load(f)
        self.assertIsInstance(cd, ConfigDict)

    def test_attributesetting(self):
        c = ConfigDict()
        try: #TODO this isn't the "proper" way to do this test
            c.a = 4,  #shoudl fail]
            self.fail('Should have raised an Assertion Error')
        except AttributeError:
            pass

        c.register('a',[])
        c.a.append(3)  #should succeed

    def test_get(self):
        c = ConfigDict().registerfromkvlist((('a', 1), ('b.c.d', 3), ('b.c.c', 2), ('b.d', 0)), namespace='')
        self.assertEqual(c.get('a'), 1)
        self.assertEqual(c.get('b.c.d'), 3)
        c.unregister('b.c.d')
        self.assertRaises(KeyError, c.get, 'b.c.d', inherit=False)
        self.assertEqual(c.get('b.c.d', default=50), 50)
        self.assertEqual(c.get('b.c.d', inherit=True), 0)

    def test_registered(self):
        c = ConfigDict().registerfromkvlist((('a', 1), ('b.c.d', 3)), namespace='')
        self.assertTrue(c.registered('a', error=False))
        self.assertTrue(c.registered('b.c.d'))
        self.assertFalse(c.registered('b.c.z'))
        self.assertRaises(KeyError, c.registered, 'b.c.z', error=True)

    def test_keyisvalid(self):
        c = ConfigDict()
        self.assertTrue(c.keyisvalid('a.b.c'))
        self.assertTrue(c.keyisvalid('a'))
        for r in RESERVED:
            self.assertFalse(c.keyisvalid('a.b.c'+r))
        self.assertFalse(c.keyisvalid('a.b.c.'))
        self.assertFalse(c.keyisvalid('.a'))

    def test_update(self):
        c = ConfigDict()
        self.assertRaises(KeyError, c.update,'a', 4)
        c.register('a', 0)
        c.update('a', 4)
        self.assertEqual(c.a, 4)

    def test_register(self):
        c = ConfigDict()
        c.register('a.b.c.d.e',[])
        self.assertEqual(c.a.b.c.d.e, [])
        c.register('a.b.c.d.e',4)
        self.assertEqual(c.a.b.c.d.e, [])
        c.register('a.b.c.d.e', 4, update=True)
        self.assertEqual(c.a.b.c.d.e, 4)
        c.register('foo.bar', 1, comment='a comment')
        self.assertEqual(c.comment('foo.bar'), 'a comment')
        c.register('baz', 1)
        self.assertIsNone(c.comment('baz'))

    def test_unregister(self):
        c = ConfigDict()
        c.register('foo.bar', 1, comment='a comment')
        c.unregister('foo.bar')
        self.assertRaises(KeyError, lambda: c.foo.bar)
        self.assertRaises(KeyError, lambda: c['foo.bar._c'])

    def test_todict(self):
        with open(defaultconfigfile(),'r') as f:
            cd = yaml.load(f)
        self.assertIsInstance(cd.todict(), dict)

    # def test_save(self):
    #     self.fail()

    # def test_registerfromconfigparser(self):
    #     self.fail()
    #
    def test_registerfromkvlist(self):
        c = ConfigDict()
        kvlist = (('a.b.c.d',1), ('a',2), ('b.c.d',3))
        c.registerfromkvlist(kvlist, namespace='')
        self.assertRaises(AttributeError, lambda: c.a.b.c.d)  #.a was replaced by the kv pair that followed it
        # clobbering the whole tree
        self.assertIs(c.a, 2)
        self.assertIs(c.b.c.d, 3)
        c = ConfigDict()
        c.registerfromkvlist(kvlist, 'z')
        self.assertIs(c.z.b.c.d, 3)


if __name__ == '__main__':
    unittest.main()