using System;
using Microsoft.VisualStudio.TestPlatform.UnitTestFramework;
using Jet.WinRT;

namespace winrt_tests
{
    [TestClass]
    public class FrameTests
    {
        [TestMethod]
        public void TestConstructors()
        {
            Frame frame = new Frame(3, 5.0);
            Assert.AreEqual(3, frame.Index);
            Assert.AreEqual(5.0, frame.TimeIntervalInSeconds);
        }
    }
}
