$("[data-tag]").click((e) => {
    const currentTag = e.target.dataset.tag;
    filterByTagName(currentTag);
  });
  
  function filterByTagName(tagName) {
    if (!tagName) {
      // 태그가 선택되지 않았을 때 모든 게시물 보여주기
      $('.post-wrapper').removeClass('hidden');
    } else {
      $('.hidden').removeClass('hidden');
      $('.post-wrapper').each((index, elem) => {
        if (!elem.hasAttribute(`data-${tagName}`)) {
          $(elem).addClass('hidden');
        }
      });
    }
  
    $(`.tag`).removeClass('selected');
    $(`.tag[data-tag=${tagName}]`).addClass('selected');
  }
  