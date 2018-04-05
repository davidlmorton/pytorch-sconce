from sconce.monitors import DataframeMonitor
import unittest


class TestCompositeMonitor(unittest.TestCase):
    def test_basics(self):
        monitor_foo = DataframeMonitor(name='foo')
        monitor_bar = DataframeMonitor(name='bar')

        composite_monitor = monitor_foo + monitor_bar
        self.assertTrue(composite_monitor.foo is monitor_foo)
        self.assertTrue(composite_monitor.bar is monitor_bar)

        composite_monitor.write(data={'a': -1, 'b': -1}, step=1)
        self.assertEqual(composite_monitor.foo.df.a.iloc[-1], -1)
        self.assertEqual(composite_monitor.foo.df.b.iloc[-1], -1)
        self.assertEqual(composite_monitor.bar.df.a.iloc[-1], -1)
        self.assertEqual(composite_monitor.bar.df.b.iloc[-1], -1)

        with self.assertRaises(RuntimeError):
            # cannot add another monitor with same name 'foo'
            composite_monitor + monitor_foo

    def test_nest_flattening(self):
        monitor_foo = DataframeMonitor(name='foo')
        monitor_bar = DataframeMonitor(name='bar')
        monitor_baz = DataframeMonitor(name='baz')

        composite = monitor_foo + monitor_bar
        nested = composite + monitor_baz

        self.assertTrue(nested.foo is monitor_foo)
        self.assertTrue(nested.bar is monitor_bar)
        self.assertTrue(nested.baz is monitor_baz)

        nested.write(data={'a': -1, 'b': -1}, step=1)
        self.assertEqual(nested.foo.df.a.iloc[-1], -1)
        self.assertEqual(nested.foo.df.b.iloc[-1], -1)
        self.assertEqual(nested.bar.df.a.iloc[-1], -1)
        self.assertEqual(nested.bar.df.b.iloc[-1], -1)
        self.assertEqual(nested.baz.df.a.iloc[-1], -1)
        self.assertEqual(nested.baz.df.b.iloc[-1], -1)
