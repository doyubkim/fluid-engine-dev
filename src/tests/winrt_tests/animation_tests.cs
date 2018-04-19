using System;
using Microsoft.VisualStudio.TestPlatform.UnitTestFramework;
using Jet.WinRT;

namespace winrt_tests
{
    [TestClass]
    public class AnimationTests
    {
        [TestMethod]
        public void TestConstructors()
        {
            Animation anim = new Animation();
        }
    }
}
