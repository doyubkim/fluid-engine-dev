using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Jet.CLR;

namespace clr_tests
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
